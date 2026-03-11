import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class GLM4Config {
  const GLM4Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    this.attentionBias = false,
    required this.headDim,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    required this.numKeyValueHeads,
    this.partialRotaryFactor = 0.5,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = true,
    this.tieWordEmbeddings = false,
    this.maxPositionEmbeddings = 32768,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final bool attentionBias;

  /// May be 0 — use `hiddenSize ~/ numAttentionHeads` in that case.
  final int headDim;
  final double rmsNormEps;
  final int vocabSize;
  final int numKeyValueHeads;
  final double partialRotaryFactor;
  final double ropeTheta;
  final bool ropeTraditional;
  final bool tieWordEmbeddings;
  final int maxPositionEmbeddings;

  int get effectiveHeadDim =>
      headDim > 0 ? headDim : hiddenSize ~/ numAttentionHeads;

  int get ropeDims =>
      (partialRotaryFactor * effectiveHeadDim).round();

  factory GLM4Config.fromJson(Map<String, dynamic> j) => GLM4Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        attentionBias: j['attention_bias'] as bool? ?? false,
        headDim: j['head_dim'] as int? ?? 0,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 0.5,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeTraditional: j['rope_traditional'] as bool? ?? true,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 32768,
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
// Attention — partial RoPE via initializeRope, optional bias
// ---------------------------------------------------------------------------

final class _GLM4Attention extends Module {
  _GLM4Attention(MLXContext ctx, GLM4Config cfg)
      : _numHeads = cfg.numAttentionHeads,
        _numKVHeads = cfg.numKeyValueHeads,
        _headDim = cfg.effectiveHeadDim,
        _scale = math.pow(cfg.effectiveHeadDim, -0.5).toDouble(),
        qProj = Linear(ctx,
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
            bias: false),
        rope = initializeRope(
          ctx,
          dims: cfg.ropeDims,
          base: cfg.ropeTheta,
          traditional: cfg.ropeTraditional,
          scalingConfig: null,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  final int _numHeads;
  final int _numKVHeads;
  final int _headDim;
  final double _scale;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = qProj.call(x)
        .reshape([b, s, _numHeads, _headDim])
        .transpose([0, 2, 1, 3]);
    var k = kProj.call(x)
        .reshape([b, s, _numKVHeads, _headDim])
        .transpose([0, 2, 1, 3]);
    final v = vProj.call(x)
        .reshape([b, s, _numKVHeads, _headDim])
        .transpose([0, 2, 1, 3]);

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
      scale: _scale,
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
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, _numHeads * _headDim]);
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
// MLP — fused gate_up_proj (2×intermediate), SiLU-gated
// ---------------------------------------------------------------------------

final class _GLM4MLP extends Module {
  _GLM4MLP(MLXContext ctx, GLM4Config cfg)
      : gateUpProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: 2 * cfg.intermediateSize,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: false),
        _intermediateSize = cfg.intermediateSize;

  Linear gateUpProj;
  Linear downProj;
  final int _intermediateSize;

  MLXArray call(MLXArray x) {
    final gateUp = gateUpProj.call(x); // [B, L, 2*I]
    final gate = gateUp.slice(
        start: [0, 0, 0], stop: [gateUp.dim(0), gateUp.dim(1), _intermediateSize]);
    final up = gateUp.slice(
        start: [0, 0, _intermediateSize],
        stop: [gateUp.dim(0), gateUp.dim(1), 2 * _intermediateSize]);
    gateUp.dispose();
    final inner = gate.silu() * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(inner);
    inner.dispose();
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
// Decoder layer — 4 norms, pre+post for both attn and MLP
//
// x = x + postSelfAttnNorm(attn(inputNorm(x)))
// x = postMlpNorm(mlp(postAttnNorm(x))) + x
// ---------------------------------------------------------------------------

final class _GLM4DecoderLayer extends Module {
  _GLM4DecoderLayer(MLXContext ctx, GLM4Config cfg)
      : selfAttn = _GLM4Attention(ctx, cfg),
        mlp = _GLM4MLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postSelfAttnLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postMlpLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _GLM4Attention selfAttn;
  _GLM4MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postSelfAttnLayernorm;
  RMSNorm postAttentionLayernorm;
  RMSNorm postMlpLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    // Pre-norm → attention → post-attn-norm → residual
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final postAttn = postSelfAttnLayernorm.call(attnOut);
    attnOut.dispose();
    final h = x + postAttn;
    postAttn.dispose();

    // Pre-norm → MLP → post-mlp-norm → residual
    final normed2 = postAttentionLayernorm.call(h);
    final mlpOut = mlp.call(normed2);
    normed2.dispose();
    final postMlp = postMlpLayernorm.call(mlpOut);
    mlpOut.dispose();
    final out = h + postMlp;
    postMlp.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postSelfAttnLayernorm.parameters().entries)
          'post_self_attn_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
        for (final e in postMlpLayernorm.parameters().entries)
          'post_mlp_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postSelfAttnLayernorm
        .loadWeights(_scoped(weights, 'post_self_attn_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    postMlpLayernorm.loadWeights(_scoped(weights, 'post_mlp_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
    postSelfAttnLayernorm.dispose();
    postAttentionLayernorm.dispose();
    postMlpLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _GLM4InnerModel extends Module {
  _GLM4InnerModel(MLXContext ctx, GLM4Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _GLM4DecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_GLM4DecoderLayer> layers;
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
// Public GLM4Model
// ---------------------------------------------------------------------------

final class GLM4Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  GLM4Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _GLM4InnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final GLM4Config config;
  final _GLM4InnerModel _model;
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
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    if (!config.tieWordEmbeddings) return weights;
    // When tied, lm_head.weight is not present in checkpoint — share embed weight
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
