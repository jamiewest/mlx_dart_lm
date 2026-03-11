import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class GemmaConfig {
  const GemmaConfig({
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

  int get effectiveHeadDim => headDim ?? hiddenSize ~/ numAttentionHeads;

  factory GemmaConfig.fromJson(Map<String, dynamic> j) => GemmaConfig(
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
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 8192,
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

MLXArray _additiveCausalMask(
  MLXContext ctx, {
  required int n,
  required int offset,
  required MLXDtype dtype,
}) {
  final boolMask = createCausalMask(ctx, n: n, offset: offset);
  final zeros = MLXArray.zeros(ctx, [1], dtype: dtype);
  final negLarge = MLXArray.fromFloats(ctx, [-1e9]).astype(dtype);
  final result = where(ctx, boolMask, zeros, negLarge);
  boolMask.dispose();
  zeros.dispose();
  negLarge.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// GemmaRMSNorm — weight shift by +1
// ---------------------------------------------------------------------------

/// RMS normalisation with weight initialised to zeros and applied as (1+w).
///
/// Mirrors `GemmaRMSNorm` from mlx-swift-lm.
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
// MLP
// ---------------------------------------------------------------------------

final class _GemmaMLP extends Module {
  _GemmaMLP(MLXContext ctx, GemmaConfig cfg)
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
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
// Attention
// ---------------------------------------------------------------------------

final class _GemmaAttention extends Module {
  _GemmaAttention(MLXContext ctx, GemmaConfig cfg)
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
        scale = 1.0 / math.sqrt(cfg.effectiveHeadDim.toDouble()),
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
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k = kProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = vProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    q = rope.call(q, offset);
    k = rope.call(k, offset);

    MLXArray? mask;
    if (s > 1) {
      mask = _additiveCausalMask(ctx, n: s, offset: offset, dtype: q.dtype);
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
        ...qProj.parameters().map((k, v) => MapEntry('q_proj.$k', v)),
        ...kProj.parameters().map((k, v) => MapEntry('k_proj.$k', v)),
        ...vProj.parameters().map((k, v) => MapEntry('v_proj.$k', v)),
        ...oProj.parameters().map((k, v) => MapEntry('o_proj.$k', v)),
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
// Decoder layer
// ---------------------------------------------------------------------------

final class _GemmaDecoderLayer extends Module {
  _GemmaDecoderLayer(MLXContext ctx, GemmaConfig cfg)
      : selfAttn = _GemmaAttention(ctx, cfg),
        mlp = _GemmaMLP(ctx, cfg),
        inputLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _GemmaAttention selfAttn;
  _GemmaMLP mlp;
  _GemmaRMSNorm inputLayernorm;
  _GemmaRMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    final normed2 = postAttentionLayernorm.call(afterAttn);
    final mlpOut = mlp.call(normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...selfAttn.parameters().map((k, v) => MapEntry('self_attn.$k', v)),
        ...mlp.parameters().map((k, v) => MapEntry('mlp.$k', v)),
        ...inputLayernorm
            .parameters()
            .map((k, v) => MapEntry('input_layernorm.$k', v)),
        ...postAttentionLayernorm
            .parameters()
            .map((k, v) => MapEntry('post_attention_layernorm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _GemmaInnerModel extends Module {
  _GemmaInnerModel(MLXContext ctx, GemmaConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _GemmaDecoderLayer(ctx, cfg)),
        norm =
            _GemmaRMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _hiddenSize = cfg.hiddenSize;

  Embedding embedTokens;
  List<_GemmaDecoderLayer> layers;
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
      ...embedTokens.parameters().map((k, v) => MapEntry('embed_tokens.$k', v)),
    };
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['layers.$i.${e.key}'] = e.value;
      }
    }
    result.addAll(norm.parameters().map((k, v) => MapEntry('norm.$k', v)));
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
// Public GemmaModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Gemma-family language model with tied embeddings and GemmaRMSNorm.
///
/// Mirrors `GemmaModel` from mlx-swift-lm.
final class GemmaModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  GemmaModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _GemmaInnerModel(ctx, config);

  final MLXContext _ctx;
  final GemmaConfig config;
  final _GemmaInnerModel _model;

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
    final logits = h.matmul(_model.embedTokens.weight.T);
    h.dispose();
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
  Map<String, MLXArray> parameters() =>
      _model.parameters().map((k, v) => MapEntry('model.$k', v));

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
  }

  @override
  void dispose() => _model.dispose();
}
