import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Gemma3TextConfig {
  const Gemma3TextConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    this.rmsNormEps = 1e-6,
    required this.vocabSize,
    this.ropeTheta = 1000000.0,
    this.ropeLocalBaseFreq = 10000.0,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.queryPreAttnScalar = 256.0,
    this.slidingWindow = 512,
    this.slidingWindowPattern = 6,
    this.maxPositionEmbeddings = 32768,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final double ropeLocalBaseFreq;
  final bool ropeTraditional;
  final Map<String, dynamic>? ropeScaling;
  final double queryPreAttnScalar;
  final int slidingWindow;
  final int slidingWindowPattern;
  final int maxPositionEmbeddings;

  double get attnScale => 1.0 / math.sqrt(queryPreAttnScalar);

  factory Gemma3TextConfig.fromJson(Map<String, dynamic> j) {
    // Support weights nested under text_config (VLM conversion).
    final root = (j['text_config'] as Map<String, dynamic>?) ?? j;
    return Gemma3TextConfig(
      hiddenSize: root['hidden_size'] as int? ?? 1152,
      numHiddenLayers: root['num_hidden_layers'] as int? ?? 26,
      intermediateSize: root['intermediate_size'] as int? ?? 6912,
      numAttentionHeads: root['num_attention_heads'] as int? ?? 4,
      numKeyValueHeads:
          root['num_key_value_heads'] as int? ?? root['num_attention_heads'] as int? ?? 4,
      headDim: root['head_dim'] as int? ?? 256,
      rmsNormEps: (root['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
      vocabSize: root['vocab_size'] as int? ?? 262144,
      ropeTheta: (root['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
      ropeLocalBaseFreq:
          (root['rope_local_base_freq'] as num?)?.toDouble() ?? 10000.0,
      ropeTraditional: root['rope_traditional'] as bool? ?? false,
      ropeScaling: root['rope_scaling'] as Map<String, dynamic>?,
      queryPreAttnScalar:
          (root['query_pre_attn_scalar'] as num?)?.toDouble() ?? 256.0,
      slidingWindow: root['sliding_window'] as int? ?? 512,
      slidingWindowPattern: root['sliding_window_pattern'] as int? ?? 6,
      maxPositionEmbeddings:
          root['max_position_embeddings'] as int? ?? 32768,
    );
  }
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

final class _Gemma3MLP extends Module {
  _Gemma3MLP(MLXContext ctx,
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
// Attention — QK norms + two RoPE bases (local vs global)
// ---------------------------------------------------------------------------

final class _Gemma3Attention extends Module {
  _Gemma3Attention(MLXContext ctx, Gemma3TextConfig cfg,
      {required this.isSliding})
      : qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: false),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        qNorm = _GemmaRMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
        kNorm = _GemmaRMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = cfg.attnScale,
        slidingWindow = isSliding ? cfg.slidingWindow : null,
        // Sliding layers use a local base frequency; global layers use the
        // full ropeTheta with the rope_scaling config.
        rope = isSliding
            ? DefaultRope(
                dims: cfg.headDim,
                traditional: false,
                base: cfg.ropeLocalBaseFreq,
              )
            : initializeRope(
                ctx,
                dims: cfg.headDim,
                base: cfg.ropeTheta,
                traditional: cfg.ropeTraditional,
                scalingConfig: cfg.ropeScaling,
                maxPositionEmbeddings: cfg.maxPositionEmbeddings,
              );

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  _GemmaRMSNorm qNorm;
  _GemmaRMSNorm kNorm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
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

    // QK norms applied before RoPE.
    final qn = qNorm.call(q);
    q.dispose();
    q = qn;
    final kn = kNorm.call(k);
    k.dispose();
    k = kn;

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

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: q,
      keys: fullK,
      values: fullV,
      scale: scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    mask?.dispose();
    q.dispose();
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
        for (final e in qNorm.parameters().entries) 'q_norm.${e.key}': e.value,
        for (final e in kNorm.parameters().entries) 'k_norm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    oProj.loadWeights(_scoped(weights, 'o_proj'));
    qNorm.loadWeights(_scoped(weights, 'q_norm'));
    kNorm.loadWeights(_scoped(weights, 'k_norm'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    oProj.dispose();
    qNorm.dispose();
    kNorm.dispose();
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer — 4 norms (same pattern as Gemma2)
// ---------------------------------------------------------------------------

final class _Gemma3DecoderLayer extends Module {
  _Gemma3DecoderLayer(MLXContext ctx, Gemma3TextConfig cfg, int layerIdx)
      : selfAttn = _Gemma3Attention(ctx, cfg,
            isSliding: (layerIdx + 1) % cfg.slidingWindowPattern != 0),
        mlp = _Gemma3MLP(ctx,
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

  _Gemma3Attention selfAttn;
  _Gemma3MLP mlp;
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

final class _Gemma3InnerModel extends Module {
  _Gemma3InnerModel(MLXContext ctx, Gemma3TextConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (i) => _Gemma3DecoderLayer(ctx, cfg, i)),
        norm = _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _hiddenSize = cfg.hiddenSize;

  Embedding embedTokens;
  List<_Gemma3DecoderLayer> layers;
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
// Public Gemma3TextModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Gemma 3 text language model.
///
/// Key differences from Gemma 2: QK norms on attention heads, two RoPE
/// bases (local frequency for sliding layers, global frequency for non-sliding),
/// and a separate lm_head (not tied to embeddings by default).
///
/// Mirrors `Gemma3TextModel` from mlx-swift-lm.
final class Gemma3TextModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Gemma3TextModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Gemma3InnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final Gemma3TextConfig config;
  final _Gemma3InnerModel _model;
  final Linear _lmHead;

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
    final logits = _lmHead.call(h);
    h.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    final maxSize = parameters?.maxKVSize;
    return List.generate(config.numHiddenLayers, (i) {
      final isGlobal =
          (i + 1) % config.slidingWindowPattern == 0;
      if (isGlobal) {
        // Global layers use a standard (unbounded) cache.
        return maxSize != null
            ? RotatingKVCache(maxSize: maxSize)
            : KVCacheSimple();
      } else {
        // Sliding-window layers use a rotating cache bounded to the window.
        return RotatingKVCache(maxSize: config.slidingWindow);
      }
    });
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  // -------------------------------------------------------------------------
  // Weight loading
  // -------------------------------------------------------------------------

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    // VLM-converted models may have weights under a language_model prefix.
    final lmPrefixed = weights.keys.any((k) => k.startsWith('language_model.'));
    if (lmPrefixed) {
      return {
        for (final e in weights.entries)
          if (e.key.startsWith('language_model.'))
            e.key.substring('language_model.'.length): e.value
          else
            e.key: e.value,
      };
    }
    // Tie lm_head to embed_tokens if missing.
    if (!weights.containsKey('lm_head.weight') &&
        weights.containsKey('model.embed_tokens.weight')) {
      return {
        ...weights,
        'lm_head.weight': weights['model.embed_tokens.weight']!,
      };
    }
    return weights;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries) 'model.${e.key}': e.value,
        for (final e in _lmHead.parameters().entries)
          'lm_head.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
    _lmHead.loadWeights(_scoped(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _model.dispose();
    _lmHead.dispose();
  }
}
