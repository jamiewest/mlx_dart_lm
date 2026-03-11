import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class PhiConfig {
  const PhiConfig({
    this.maxPositionEmbeddings = 2048,
    required this.vocabSize,
    required this.hiddenSize,
    required this.numAttentionHeads,
    required this.numHiddenLayers,
    required this.numKeyValueHeads,
    this.partialRotaryFactor = 0.4,
    required this.intermediateSize,
    this.layerNormEps = 1e-5,
    this.ropeTheta = 10000.0,
  });

  final int maxPositionEmbeddings;
  final int vocabSize;
  final int hiddenSize;
  final int numAttentionHeads;
  final int numHiddenLayers;
  final int numKeyValueHeads;
  final double partialRotaryFactor;
  final int intermediateSize;
  final double layerNormEps;
  final double ropeTheta;

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get ropeDims => (partialRotaryFactor * headDim).round();

  factory PhiConfig.fromJson(Map<String, dynamic> j) => PhiConfig(
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 2048,
        vocabSize: j['vocab_size'] as int,
        hiddenSize: j['hidden_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 0.4,
        intermediateSize: j['intermediate_size'] as int,
        layerNormEps: (j['layer_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
      );
}

// ---------------------------------------------------------------------------
// Helpers
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
// Attention — partial RoPE, biased projections, output key "dense"
// ---------------------------------------------------------------------------

final class _PhiAttention extends Module {
  _PhiAttention(MLXContext ctx, PhiConfig cfg)
      : numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        ropeDims = cfg.ropeDims,
        scale = math.sqrt(1.0 / cfg.headDim.toDouble()),
        qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: true),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: true),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: true),
        dense = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: true),
        rope = DefaultRope(
            dims: cfg.ropeDims,
            traditional: false,
            base: cfg.ropeTheta);

  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final int ropeDims;
  final double scale;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear dense;
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    final q =
        qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final k =
        kProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v =
        vProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    final qRoped = rope.call(q, offset);
    q.dispose();
    final kRoped = rope.call(k, offset);
    k.dispose();

    MLXArray? mask;
    if (s > 1) {
      mask = _additiveCausalMask(ctx,
          n: s, offset: offset, dtype: qRoped.dtype);
    }

    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(kRoped, v);
    } else {
      fullK = kRoped;
      fullV = v;
    }

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: qRoped,
      keys: fullK,
      values: fullV,
      scale: scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    mask?.dispose();
    qRoped.dispose();
    if (cache == null) {
      kRoped.dispose();
      v.dispose();
    }

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = dense.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in qProj.parameters().entries) 'q_proj.${e.key}': e.value,
        for (final e in kProj.parameters().entries) 'k_proj.${e.key}': e.value,
        for (final e in vProj.parameters().entries) 'v_proj.${e.key}': e.value,
        for (final e in dense.parameters().entries) 'dense.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    dense.loadWeights(_scoped(weights, 'dense'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    dense.dispose();
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// MLP — fc1 → GELU → fc2 (no gating)
// ---------------------------------------------------------------------------

final class _PhiMLP extends Module {
  _PhiMLP(MLXContext ctx, PhiConfig cfg)
      : fc1 = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: true),
        fc2 = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: true);

  Linear fc1;
  Linear fc2;

  MLXArray call(MLXArray x) {
    final h = fc1.call(x).gelu();
    final out = fc2.call(h);
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in fc1.parameters().entries) 'fc1.${e.key}': e.value,
        for (final e in fc2.parameters().entries) 'fc2.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    fc1.loadWeights(_scoped(weights, 'fc1'));
    fc2.loadWeights(_scoped(weights, 'fc2'));
  }

  @override
  void dispose() {
    fc1.dispose();
    fc2.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer — parallel attn+MLP from same normed input
// ---------------------------------------------------------------------------

final class _PhiDecoderLayer extends Module {
  _PhiDecoderLayer(MLXContext ctx, PhiConfig cfg)
      : selfAttn = _PhiAttention(ctx, cfg),
        mlp = _PhiMLP(ctx, cfg),
        inputLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.layerNormEps);

  _PhiAttention selfAttn;
  _PhiMLP mlp;
  LayerNorm inputLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    // Both attention and MLP receive the same pre-norm input
    final h = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, h, cache);
    final mlpOut = mlp.call(h);
    h.dispose();
    // Residual: x + attn + mlp
    final out = x + attnOut + mlpOut;
    attnOut.dispose();
    mlpOut.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _PhiInnerModel extends Module {
  _PhiInnerModel(MLXContext ctx, PhiConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _PhiDecoderLayer(ctx, cfg)),
        finalLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.layerNormEps);

  Embedding embedTokens;
  List<_PhiDecoderLayer> layers;
  LayerNorm finalLayernorm;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
    for (var i = 0; i < layers.length; i++) {
      final cache = (caches != null && i < caches.length) ? caches[i] : null;
      final next = layers[i].call(ctx, h, cache);
      h.dispose();
      h = next;
    }
    final normed = finalLayernorm.call(h);
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
    result.addAll(finalLayernorm.parameters()
        .map((k, v) => MapEntry('final_layernorm.$k', v)));
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    embedTokens.loadWeights(_scoped(weights, 'embed_tokens'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'layers.$i'));
    }
    finalLayernorm.loadWeights(_scoped(weights, 'final_layernorm'));
  }

  @override
  void dispose() {
    embedTokens.dispose();
    for (final l in layers) {
      l.dispose();
    }
    finalLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public PhiModel
// ---------------------------------------------------------------------------

final class PhiModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  PhiModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _PhiInnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: true);

  final MLXContext _ctx;
  final PhiConfig config;
  final _PhiInnerModel _model;
  final Linear _lmHead;

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
    return List.generate(config.numHiddenLayers, (_) {
      if (maxSize != null) return RotatingKVCache(maxSize: maxSize);
      return KVCacheSimple();
    });
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) =>
      weights;

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries)
          'model.${e.key}': e.value,
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
