import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Starcoder2Config {
  const Starcoder2Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.normEpsilon = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 100000.0,
    this.maxPositionEmbeddings = 16384,
    this.tieWordEmbeddings = true,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final double normEpsilon;
  final int vocabSize;
  final double ropeTheta;
  final int maxPositionEmbeddings;
  final bool tieWordEmbeddings;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  factory Starcoder2Config.fromJson(Map<String, dynamic> j) => Starcoder2Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        normEpsilon: (j['norm_epsilon'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int? ?? 49152,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 100000.0,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 16384,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? true,
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
// MLP — c_fc → GELU → c_proj (no gate projection)
// ---------------------------------------------------------------------------

final class _Starcoder2MLP extends Module {
  _Starcoder2MLP(MLXContext ctx, Starcoder2Config cfg)
      : cFc = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: true),
        cProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: true);

  Linear cFc;
  Linear cProj;

  MLXArray call(MLXArray x) {
    final activated = cFc.call(x).gelu();
    final out = cProj.call(activated);
    activated.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in cFc.parameters().entries) 'c_fc.${e.key}': e.value,
        for (final e in cProj.parameters().entries) 'c_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    cFc.loadWeights(_scoped(weights, 'c_fc'));
    cProj.loadWeights(_scoped(weights, 'c_proj'));
  }

  @override
  void dispose() {
    cFc.dispose();
    cProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

final class _Starcoder2Attention extends Module {
  _Starcoder2Attention(MLXContext ctx, Starcoder2Config cfg)
      : qProj = Linear(ctx,
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
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: true),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _ropeBase = cfg.ropeTheta;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final double _ropeBase;

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
    q = q.rope(dims: headDim, traditional: false, base: _ropeBase, offset: offset);
    k = k.rope(dims: headDim, traditional: false, base: _ropeBase, offset: offset);

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
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _Starcoder2DecoderLayer extends Module {
  _Starcoder2DecoderLayer(MLXContext ctx, Starcoder2Config cfg)
      : selfAttn = _Starcoder2Attention(ctx, cfg),
        mlp = _Starcoder2MLP(ctx, cfg),
        inputLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEpsilon),
        postAttentionLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEpsilon);

  _Starcoder2Attention selfAttn;
  _Starcoder2MLP mlp;
  LayerNorm inputLayernorm;
  LayerNorm postAttentionLayernorm;

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

final class _Starcoder2InnerModel extends Module {
  _Starcoder2InnerModel(MLXContext ctx, Starcoder2Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _Starcoder2DecoderLayer(ctx, cfg)),
        norm = LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEpsilon);

  Embedding embedTokens;
  List<_Starcoder2DecoderLayer> layers;
  LayerNorm norm;

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
// Public Starcoder2Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// Starcoder 2 language model.
///
/// Uses LayerNorm, biased attention projections, and a simple 2-layer MLP
/// (no gate projection). Supports tied/untied embeddings.
///
/// Mirrors `Starcoder2Model` from mlx-swift-lm.
final class Starcoder2Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Starcoder2Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Starcoder2InnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Starcoder2Config config;
  final _Starcoder2InnerModel _model;
  final Linear? _lmHead;

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
    final logits = _lmHead != null
        ? _lmHead.call(h)
        : h.matmul(_model.embedTokens.weight.T);
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
    if (config.tieWordEmbeddings &&
        !weights.containsKey('lm_head.weight') &&
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
        if (_lmHead != null)
          for (final e in _lmHead.parameters().entries)
            'lm_head.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
    _lmHead?.loadWeights(_scoped(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _model.dispose();
    _lmHead?.dispose();
  }
}
