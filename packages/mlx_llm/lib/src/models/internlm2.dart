import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Internlm2Config {
  const Internlm2Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 1000000.0,
    this.ropeScaling,
    this.maxPositionEmbeddings = 32768,
    this.tieWordEmbeddings = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final int maxPositionEmbeddings;
  final bool tieWordEmbeddings;

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get kvGroups => numAttentionHeads ~/ numKeyValueHeads;

  factory Internlm2Config.fromJson(Map<String, dynamic> j) => Internlm2Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 32768,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
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
// MLP — w1 (gate) · w3 (up) → silu → w2 (down)
// ---------------------------------------------------------------------------

final class _Internlm2MLP extends Module {
  _Internlm2MLP(MLXContext ctx, Internlm2Config cfg)
      : w1 = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false),
        w2 = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: false),
        w3 = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false);

  Linear w1;
  Linear w2;
  Linear w3;

  MLXArray call(MLXArray x) {
    final gate = w1.call(x).silu();
    final up = w3.call(x);
    final inner = gate * up;
    gate.dispose();
    up.dispose();
    final out = w2.call(inner);
    inner.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in w1.parameters().entries) 'w1.${e.key}': e.value,
        for (final e in w2.parameters().entries) 'w2.${e.key}': e.value,
        for (final e in w3.parameters().entries) 'w3.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    w1.loadWeights(_scoped(weights, 'w1'));
    w2.loadWeights(_scoped(weights, 'w2'));
    w3.loadWeights(_scoped(weights, 'w3'));
  }

  @override
  void dispose() {
    w1.dispose();
    w2.dispose();
    w3.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — fused wqkv
// ---------------------------------------------------------------------------

final class _Internlm2Attention extends Module {
  _Internlm2Attention(MLXContext ctx, Internlm2Config cfg)
      : wqkv = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures:
                (cfg.numAttentionHeads + 2 * cfg.numKeyValueHeads) * cfg.headDim,
            bias: false),
        wo = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        kvGroups = cfg.kvGroups,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = initializeRope(
          ctx,
          dims: cfg.headDim,
          traditional: false,
          base: cfg.ropeTheta,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  Linear wqkv;
  Linear wo;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final int kvGroups;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Fused QKV projection: [B, L, (numHeads + 2*numKVHeads)*headDim]
    // Reshape to [B, L, -1, 2+kvGroups, headDim] then slice
    final qkvRaw = wqkv.call(x);
    final qkvReshaped =
        qkvRaw.reshape([b, s, numKVHeads, 2 + kvGroups, headDim]);
    qkvRaw.dispose();

    // Q: first kvGroups slabs along axis 3 → shape [B, L, numKVHeads, kvGroups, headDim]
    // then merge → [B, L, numHeads, headDim]
    final qRaw = qkvReshaped.slice(
      start: [0, 0, 0, 0, 0],
      stop: [b, s, numKVHeads, kvGroups, headDim],
    );
    final q = qRaw
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    qRaw.dispose();

    // K: second-to-last slab → [B, L, numKVHeads, 1, headDim] → [B, L, numKVHeads, headDim]
    final kRaw = qkvReshaped.slice(
      start: [0, 0, 0, kvGroups, 0],
      stop: [b, s, numKVHeads, kvGroups + 1, headDim],
    );
    final k = kRaw
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    kRaw.dispose();

    // V: last slab → [B, L, numKVHeads, 1, headDim] → [B, L, numKVHeads, headDim]
    final vRaw = qkvReshaped.slice(
      start: [0, 0, 0, kvGroups + 1, 0],
      stop: [b, s, numKVHeads, kvGroups + 2, headDim],
    );
    final v = vRaw
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    vRaw.dispose();
    qkvReshaped.dispose();

    final offset = cache?.offset ?? 0;
    final qRoped = _rope.call(q, offset);
    q.dispose();
    final kRoped = _rope.call(k, offset);
    k.dispose();

    MLXArray? mask;
    if (s > 1) {
      mask = createCausalMask(ctx, n: s, offset: offset);
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
      maskMode: mask != null ? 'causal' : 'none',
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
    final out = wo.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in wqkv.parameters().entries) 'wqkv.${e.key}': e.value,
        for (final e in wo.parameters().entries) 'wo.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    wqkv.loadWeights(_scoped(weights, 'wqkv'));
    wo.loadWeights(_scoped(weights, 'wo'));
  }

  @override
  void dispose() {
    wqkv.dispose();
    wo.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _Internlm2DecoderLayer extends Module {
  _Internlm2DecoderLayer(MLXContext ctx, Internlm2Config cfg)
      : attention = _Internlm2Attention(ctx, cfg),
        feedForward = _Internlm2MLP(ctx, cfg),
        attentionNorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        ffnNorm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Internlm2Attention attention;
  _Internlm2MLP feedForward;
  RMSNorm attentionNorm;
  RMSNorm ffnNorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = attentionNorm.call(x);
    final attnOut = attention.call(ctx, normed, cache);
    normed.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    final normed2 = ffnNorm.call(afterAttn);
    final mlpOut = feedForward.call(normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attention.parameters().entries)
          'attention.${e.key}': e.value,
        for (final e in feedForward.parameters().entries)
          'feed_forward.${e.key}': e.value,
        for (final e in attentionNorm.parameters().entries)
          'attention_norm.${e.key}': e.value,
        for (final e in ffnNorm.parameters().entries)
          'ffn_norm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention.loadWeights(_scoped(weights, 'attention'));
    feedForward.loadWeights(_scoped(weights, 'feed_forward'));
    attentionNorm.loadWeights(_scoped(weights, 'attention_norm'));
    ffnNorm.loadWeights(_scoped(weights, 'ffn_norm'));
  }

  @override
  void dispose() {
    attention.dispose();
    feedForward.dispose();
    attentionNorm.dispose();
    ffnNorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _Internlm2InnerModel extends Module {
  _Internlm2InnerModel(MLXContext ctx, Internlm2Config cfg)
      : tokEmbeddings =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _Internlm2DecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding tokEmbeddings;
  List<_Internlm2DecoderLayer> layers;
  RMSNorm norm;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = tokEmbeddings.call(tokens);
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
      for (final e in tokEmbeddings.parameters().entries)
        'tok_embeddings.${e.key}': e.value,
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
    tokEmbeddings.loadWeights(_scoped(weights, 'tok_embeddings'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'layers.$i'));
    }
    norm.loadWeights(_scoped(weights, 'norm'));
  }

  @override
  void dispose() {
    tokEmbeddings.dispose();
    for (final l in layers) {
      l.dispose();
    }
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public Internlm2Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// InternLM2 language model.
///
/// Uses a fused `wqkv` projection reshaped to extract Q/K/V via tensor slicing,
/// SiLU-gated MLP (`w1`/`w2`/`w3`), and non-standard weight-key conventions
/// (`tok_embeddings`, `attention`, `feed_forward`, `output`).
///
/// Mirrors `InternLM2Model` from mlx-swift-lm.
final class Internlm2Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Internlm2Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Internlm2InnerModel(ctx, config),
        _output = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Internlm2Config config;
  final _Internlm2InnerModel _model;
  final Linear? _output;

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
    final logits = _output != null
        ? _output.call(h)
        : h.matmul(_model.tokEmbeddings.weight.T);
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
    // Remove rope inv_freq buffers — not needed at inference time
    return {
      for (final e in weights.entries)
        if (!e.key.contains('attention.rope.inv_freq')) e.key: e.value,
    };
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries) 'model.${e.key}': e.value,
        if (_output != null)
          for (final e in _output.parameters().entries)
            'output.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
    _output?.loadWeights(_scoped(weights, 'output'));
  }

  @override
  void dispose() {
    _model.dispose();
    _output?.dispose();
  }
}
