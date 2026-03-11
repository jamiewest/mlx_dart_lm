import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Gemma2Config {
  const Gemma2Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.headDim,
    this.rmsNormEps = 1e-6,
    required this.vocabSize,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.attentionBias = false,
    this.maxPositionEmbeddings = 8192,
    this.slidingWindow = 4096,
    this.slidingWindowPattern = 2,
    this.queryPreAttnScalar,
    this.attnLogitSoftcapping = 50.0,
    this.finalLogitSoftcapping = 30.0,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int? headDim;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool ropeTraditional;
  final Map<String, dynamic>? ropeScaling;
  final bool attentionBias;
  final int maxPositionEmbeddings;
  final int slidingWindow;
  final int slidingWindowPattern;
  final int? queryPreAttnScalar;
  final double attnLogitSoftcapping;
  final double finalLogitSoftcapping;

  int get effectiveHeadDim => headDim ?? hiddenSize ~/ numAttentionHeads;

  double get attnScale {
    final scalar = queryPreAttnScalar;
    if (scalar != null) return 1.0 / math.sqrt(scalar.toDouble());
    return 1.0 / math.sqrt(effectiveHeadDim.toDouble());
  }

  factory Gemma2Config.fromJson(Map<String, dynamic> j) => Gemma2Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        headDim: j['head_dim'] as int?,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeTraditional: j['rope_traditional'] as bool? ?? false,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        attentionBias: j['attention_bias'] as bool? ?? false,
        maxPositionEmbeddings: j['max_position_embeddings'] as int? ?? 8192,
        slidingWindow: j['sliding_window'] as int? ?? 4096,
        slidingWindowPattern: j['sliding_window_pattern'] as int? ?? 2,
        queryPreAttnScalar: j['query_pre_attn_scalar'] as int?,
        attnLogitSoftcapping:
            (j['attn_logit_softcapping'] as num?)?.toDouble() ?? 50.0,
        finalLogitSoftcapping:
            (j['final_logit_softcapping'] as num?)?.toDouble() ?? 30.0,
      );
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

Map<String, MLXArray> _scoped(Map<String, MLXArray> weights, String prefix) {
  final p = '$prefix.';
  return {
    for (final e in weights.entries)
      if (e.key.startsWith(p)) e.key.substring(p.length): e.value,
  };
}

// ---------------------------------------------------------------------------
// GemmaRMSNorm — weight shift by +1
// ---------------------------------------------------------------------------

final class _GemmaRMSNorm extends Module {
  _GemmaRMSNorm(this.ctx, {required int dims, this.eps = 1e-6})
      : weight = MLXArray.zeros(ctx, [dims]);

  final MLXContext ctx;
  MLXArray weight;
  final double eps;

  MLXArray call(MLXArray x) {
    final ones = MLXArray.ones(ctx, weight.shape);
    final shifted = ones + weight;
    ones.dispose();
    final result = x.rmsNorm(shifted, eps: eps);
    shifted.dispose();
    return result;
  }

  @override
  Map<String, MLXArray> parameters() => {'weight': weight};

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) weight = w;
  }

  @override
  void dispose() => weight.dispose();
}

// ---------------------------------------------------------------------------
// MLP (GELU activation)
// ---------------------------------------------------------------------------

final class _Gemma2MLP extends Module {
  _Gemma2MLP(MLXContext ctx,
      {required int hiddenSize, required int intermediateSize})
      : gateProj = Linear(ctx,
            inFeatures: hiddenSize,
            outFeatures: intermediateSize,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: hiddenSize,
            outFeatures: intermediateSize,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: intermediateSize,
            outFeatures: hiddenSize,
            bias: false);

  Linear gateProj;
  Linear upProj;
  Linear downProj;

  MLXArray call(MLXArray x) {
    final gate = gateProj.call(x).gelu();
    final up = upProj.call(x);
    final gated = gate * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(gated);
    gated.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in gateProj.parameters().entries)
          'gate_proj.${e.key}': e.value,
        for (final e in upProj.parameters().entries)
          'up_proj.${e.key}': e.value,
        for (final e in downProj.parameters().entries)
          'down_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gateProj.loadWeights(_scoped(weights, 'gate_proj'));
    upProj.loadWeights(_scoped(weights, 'up_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
  }

  @override
  void dispose() {
    gateProj.dispose();
    upProj.dispose();
    downProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention with logit soft-capping
// ---------------------------------------------------------------------------

final class _Gemma2Attention extends Module {
  _Gemma2Attention(MLXContext ctx, Gemma2Config cfg,
      {required this.isSliding})
      : qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.effectiveHeadDim,
            outFeatures: cfg.hiddenSize,
            bias: cfg.attentionBias),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.effectiveHeadDim,
        scale = cfg.attnScale,
        softcap = cfg.attnLogitSoftcapping,
        slidingWindow = isSliding ? cfg.slidingWindow : null,
        rope = initializeRope(
          ctx,
          dims: cfg.effectiveHeadDim,
          base: cfg.ropeTheta,
          traditional: cfg.ropeTraditional,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final double softcap;
  final int? slidingWindow;
  final bool isSliding;
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = qProj
        .call(x)
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    var k = kProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final v = vProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    q = rope.call(q, offset);
    k = rope.call(k, offset);

    // Build additive causal mask (with optional sliding window).
    MLXArray? mask;
    if (s > 1) {
      final boolMask = createCausalMask(ctx,
          n: s, offset: offset, windowSize: slidingWindow);
      final zeros = MLXArray.zeros(ctx, [1], dtype: q.dtype);
      final negLarge = MLXArray.fromFloats(ctx, [-1e9]).astype(q.dtype);
      mask = where(ctx, boolMask, zeros, negLarge);
      boolMask.dispose();
      zeros.dispose();
      negLarge.dispose();
    }

    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(k, v);
    } else {
      fullK = k;
      fullV = v;
    }

    // Repeat K/V heads for GQA.
    MLXArray attnK = fullK;
    MLXArray attnV = fullV;
    bool ownKV = false;
    if (numHeads != numKVHeads) {
      final repeats = numHeads ~/ numKVHeads;
      attnK = fullK.repeat(repeats, axis: 1);
      attnV = fullV.repeat(repeats, axis: 1);
      ownKV = true;
    }

    // Manual attention with logit soft-capping.
    final qk = q.matmul(attnK.transpose([0, 1, 3, 2]));
    q.dispose();
    final scaleArr = MLXArray.float_(ctx, scale);
    var scores = qk * scaleArr;
    scaleArr.dispose();
    qk.dispose();

    // tanh soft cap
    final capArr = MLXArray.float_(ctx, softcap);
    final invCap = MLXArray.float_(ctx, 1.0 / softcap);
    final t = scores * invCap;
    invCap.dispose();
    scores.dispose();
    final tanhT = t.tanh();
    t.dispose();
    scores = tanhT * capArr;
    tanhT.dispose();
    capArr.dispose();

    if (mask != null) {
      final masked = scores + mask;
      scores.dispose();
      mask.dispose();
      scores = masked;
    }

    final weights = scores.softmax();
    scores.dispose();

    final attnOut = weights.matmul(attnV);
    weights.dispose();
    if (ownKV) {
      attnK.dispose();
      attnV.dispose();
    }
    if (cache == null) {
      k.dispose();
      v.dispose();
    }

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = oProj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in qProj.parameters().entries) 'q_proj.${e.key}': e.value,
        for (final e in kProj.parameters().entries) 'k_proj.${e.key}': e.value,
        for (final e in vProj.parameters().entries) 'v_proj.${e.key}': e.value,
        for (final e in oProj.parameters().entries) 'o_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    oProj.loadWeights(_scoped(weights, 'o_proj'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    oProj.dispose();
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer — 4 norms
// ---------------------------------------------------------------------------

final class _Gemma2DecoderLayer extends Module {
  _Gemma2DecoderLayer(MLXContext ctx, Gemma2Config cfg, int layerIndex)
      : selfAttn = _Gemma2Attention(ctx, cfg,
            isSliding: (layerIndex % cfg.slidingWindowPattern) !=
                (cfg.slidingWindowPattern - 1)),
        mlp = _Gemma2MLP(ctx,
            hiddenSize: cfg.hiddenSize,
            intermediateSize: cfg.intermediateSize),
        inputLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        preFeedforwardLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postFeedforwardLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Gemma2Attention selfAttn;
  _Gemma2MLP mlp;
  _GemmaRMSNorm inputLayernorm;
  _GemmaRMSNorm postAttentionLayernorm;
  _GemmaRMSNorm preFeedforwardLayernorm;
  _GemmaRMSNorm postFeedforwardLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    // Attention block: pre-norm → attn → post-norm → residual
    final normed = inputLayernorm.call(x);
    final attnRaw = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final attnNormed = postAttentionLayernorm.call(attnRaw);
    attnRaw.dispose();
    final afterAttn = x + attnNormed;
    attnNormed.dispose();

    // MLP block: pre-norm → mlp → post-norm → residual
    final normed2 = preFeedforwardLayernorm.call(afterAttn);
    final mlpRaw = mlp.call(normed2);
    normed2.dispose();
    final mlpNormed = postFeedforwardLayernorm.call(mlpRaw);
    mlpRaw.dispose();
    final out = afterAttn + mlpNormed;
    mlpNormed.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
        for (final e in preFeedforwardLayernorm.parameters().entries)
          'pre_feedforward_layernorm.${e.key}': e.value,
        for (final e in postFeedforwardLayernorm.parameters().entries)
          'post_feedforward_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    preFeedforwardLayernorm
        .loadWeights(_scoped(weights, 'pre_feedforward_layernorm'));
    postFeedforwardLayernorm
        .loadWeights(_scoped(weights, 'post_feedforward_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
    preFeedforwardLayernorm.dispose();
    postFeedforwardLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _Gemma2InnerModel extends Module {
  _Gemma2InnerModel(MLXContext ctx, Gemma2Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (i) => _Gemma2DecoderLayer(ctx, cfg, i)),
        norm = _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _hiddenSize = cfg.hiddenSize;

  Embedding embedTokens;
  List<_Gemma2DecoderLayer> layers;
  _GemmaRMSNorm norm;
  final int _hiddenSize;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);

    // Scale embedding output by sqrt(hiddenSize).
    final scaleArr = MLXArray.float_(ctx, math.sqrt(_hiddenSize.toDouble()));
    final scaled = h * scaleArr;
    scaleArr.dispose();
    h.dispose();
    h = scaled;

    for (var i = 0; i < layers.length; i++) {
      final cache = (caches != null && i < caches.length) ? caches[i] : null;
      final next = layers[i].call(ctx, h, cache);
      h.dispose();
      h = next;
    }
    final normed = norm.call(h);
    h.dispose();
    return normed;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in embedTokens.parameters().entries)
        'embed_tokens.${e.key}': e.value,
    };
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['layers.$i.${e.key}'] = e.value;
      }
    }
    for (final e in norm.parameters().entries) {
      result['norm.${e.key}'] = e.value;
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    embedTokens.loadWeights(_scoped(weights, 'embed_tokens'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'layers.$i'));
    }
    norm.loadWeights(_scoped(weights, 'norm'));
  }

  @override
  void dispose() {
    embedTokens.dispose();
    for (final l in layers) {
      l.dispose();
    }
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public Gemma2Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// Gemma 2 language model with sliding-window attention and logit soft-capping.
///
/// Mirrors `Gemma2Model` from mlx-swift-lm.
final class Gemma2Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Gemma2Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Gemma2InnerModel(ctx, config);

  final MLXContext _ctx;
  final Gemma2Config config;
  final _Gemma2InnerModel _model;

  // -------------------------------------------------------------------------
  // LanguageModel
  // -------------------------------------------------------------------------

  @override
  PrepareResult prepare(LMInput input, List<KVCache> cache,
      {int? windowSize}) {
    final tokens = input.text.tokens;
    final seqLen = tokens.dim(0);
    final chunkSize = windowSize ?? seqLen;

    LMOutput? lastOutput;
    var pos = 0;
    while (pos < seqLen) {
      final end = (pos + chunkSize).clamp(0, seqLen);
      final chunk = tokens.slice(start: [pos], stop: [end]);
      final chunkText = LMInputText(tokens: chunk.expandDims(0));
      final prev = lastOutput;
      lastOutput = call(chunkText, cache: cache, state: prev?.state);
      chunk.dispose();
      pos = end;
    }
    return PrepareResult.logits(lastOutput!);
  }

  @override
  LMOutput call(LMInputText input, {List<KVCache>? cache, LMState? state}) {
    final h = _model.call(_ctx, input.tokens, cache);
    // Tied embeddings: use embed_tokens weight transposed as lm_head.
    var logits = h.matmul(_model.embedTokens.weight.T);
    h.dispose();

    // Final logit soft-capping.
    final cap = config.finalLogitSoftcapping;
    if (cap != 0.0) {
      final capArr = MLXArray.float_(_ctx, cap);
      final invCap = MLXArray.float_(_ctx, 1.0 / cap);
      final t = logits * invCap;
      invCap.dispose();
      final tanhT = t.tanh();
      t.dispose();
      final capped = tanhT * capArr;
      tanhT.dispose();
      capArr.dispose();
      logits.dispose();
      logits = capped;
    }

    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    final maxSize = parameters?.maxKVSize;
    return List.generate(config.numHiddenLayers, (_) {
      if (maxSize != null) return RotatingKVCache(maxSize: maxSize);
      return KVCacheSimple();
    });
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  // -------------------------------------------------------------------------
  // Weight loading
  // -------------------------------------------------------------------------

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) =>
      weights;

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries) 'model.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
  }

  @override
  void dispose() => _model.dispose();
}
