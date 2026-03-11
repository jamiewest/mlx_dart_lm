import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Phi3Config {
  const Phi3Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.partialRotaryFactor = 1.0,
    this.attentionBias = false,
    this.maxPositionEmbeddings = 4096,
    this.originalMaxPositionEmbeddings,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool ropeTraditional;
  final Map<String, dynamic>? ropeScaling;
  final double partialRotaryFactor;
  final bool attentionBias;
  final int maxPositionEmbeddings;
  final int? originalMaxPositionEmbeddings;

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get ropeDim => (headDim * partialRotaryFactor).toInt();

  factory Phi3Config.fromJson(Map<String, dynamic> j) => Phi3Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeTraditional: j['rope_traditional'] as bool? ?? false,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 1.0,
        attentionBias: j['attention_bias'] as bool? ?? false,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 4096,
        originalMaxPositionEmbeddings:
            j['original_max_position_embeddings'] as int?,
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
// MLP — fused gate_up_proj
// ---------------------------------------------------------------------------

final class _Phi3MLP extends Module {
  _Phi3MLP(MLXContext ctx, Phi3Config cfg)
      : gateUpProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize * 2,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: false);

  Linear gateUpProj;
  Linear downProj;

  MLXArray call(MLXArray x) {
    final gateUp = gateUpProj.call(x);
    final parts = gateUp.split(2, axis: 2);
    gateUp.dispose();
    final gate = parts[0].silu();
    parts[0].dispose();
    final gated = gate * parts[1];
    gate.dispose();
    parts[1].dispose();
    final out = downProj.call(gated);
    gated.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in gateUpProj.parameters().entries)
          'gate_up_proj.${e.key}': e.value,
        for (final e in downProj.parameters().entries)
          'down_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gateUpProj.loadWeights(_scoped(weights, 'gate_up_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
  }

  @override
  void dispose() {
    gateUpProj.dispose();
    downProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — fused qkv_proj + partial rotary
// ---------------------------------------------------------------------------

final class _Phi3Attention extends Module {
  _Phi3Attention(MLXContext ctx, Phi3Config cfg)
      : qkvProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim +
                2 * cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        qSize = cfg.numAttentionHeads * cfg.headDim,
        kvSize = cfg.numKeyValueHeads * cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        rope = initializeRope(
          ctx,
          dims: cfg.ropeDim,
          base: cfg.ropeTheta,
          traditional: cfg.ropeTraditional,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  Linear qkvProj;
  Linear oProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final int qSize;
  final int kvSize;
  final double scale;
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Split fused qkv projection.
    final qkv = qkvProj.call(x); // [b, s, qSize + 2*kvSize]
    final qkvRank = qkv.ndim;
    // q = qkv[..., 0:qSize]
    final stopQ = List<int>.from(qkv.shape);
    stopQ[qkvRank - 1] = qSize;
    final qFlat = qkv.slice(start: List.filled(qkvRank, 0), stop: stopQ);
    // k = qkv[..., qSize:qSize+kvSize]
    final startK = List.filled(qkvRank, 0);
    startK[qkvRank - 1] = qSize;
    final stopK = List<int>.from(qkv.shape);
    stopK[qkvRank - 1] = qSize + kvSize;
    final kFlat = qkv.slice(start: startK, stop: stopK);
    // v = qkv[..., qSize+kvSize:]
    final startV = List.filled(qkvRank, 0);
    startV[qkvRank - 1] = qSize + kvSize;
    final vFlat = qkv.slice(start: startV, stop: List<int>.from(qkv.shape));
    qkv.dispose();

    var q = qFlat.reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k = kFlat.reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = vFlat.reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    qFlat.dispose();
    kFlat.dispose();
    vFlat.dispose();

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
        for (final e in qkvProj.parameters().entries)
          'qkv_proj.${e.key}': e.value,
        for (final e in oProj.parameters().entries) 'o_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qkvProj.loadWeights(_scoped(weights, 'qkv_proj'));
    oProj.loadWeights(_scoped(weights, 'o_proj'));
  }

  @override
  void dispose() {
    qkvProj.dispose();
    oProj.dispose();
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _Phi3DecoderLayer extends Module {
  _Phi3DecoderLayer(MLXContext ctx, Phi3Config cfg)
      : selfAttn = _Phi3Attention(ctx, cfg),
        mlp = _Phi3MLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Phi3Attention selfAttn;
  _Phi3MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

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
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
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

final class _Phi3InnerModel extends Module {
  _Phi3InnerModel(MLXContext ctx, Phi3Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _Phi3DecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Phi3DecoderLayer> layers;
  RMSNorm norm;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
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
// Public Phi3Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// Phi-3 family language model with fused QKV/gate-up projections and
/// partial rotary embeddings.
///
/// Mirrors `Phi3Model` from mlx-swift-lm.
final class Phi3Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Phi3Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Phi3InnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final Phi3Config config;
  final _Phi3InnerModel _model;
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
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    // If lm_head is missing and embeddings are tied, share the weight.
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
