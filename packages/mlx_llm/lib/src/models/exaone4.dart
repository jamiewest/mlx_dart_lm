import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Exaone4Config {
  const Exaone4Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    this.rmsNormEps = 1e-6,
    this.ropeTheta = 10000.0,
    this.tieWordEmbeddings = false,
    this.maxPositionEmbeddings = 131072,
    this.slidingWindow,
    this.slidingWindowPattern,
    this.ropeScaling,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final double rmsNormEps;
  final double ropeTheta;
  final bool tieWordEmbeddings;
  final int maxPositionEmbeddings;
  final int? slidingWindow;
  final String? slidingWindowPattern;
  final Map<String, dynamic>? ropeScaling;

  /// Whether layer [i] uses local/sliding attention ('L' in pattern).
  bool isLocal(int i) {
    final pattern = slidingWindowPattern;
    if (pattern == null || pattern.isEmpty) return false;
    return pattern[i % pattern.length] == 'L';
  }

  factory Exaone4Config.fromJson(Map<String, dynamic> j) => Exaone4Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads: j['num_key_value_heads'] as int,
        headDim: j['head_dim'] as int,
        vocabSize: j['vocab_size'] as int,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        maxPositionEmbeddings: j['max_position_embeddings'] as int? ?? 131072,
        slidingWindow: j['sliding_window'] as int?,
        slidingWindowPattern: j['sliding_window_pattern'] as String?,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
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
// MLP
// ---------------------------------------------------------------------------

final class _Exaone4MLP extends Module {
  _Exaone4MLP(MLXContext ctx, Exaone4Config cfg)
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false);

  Linear gateProj;
  Linear downProj;
  Linear upProj;

  MLXArray call(MLXArray x) {
    final gate = gateProj.call(x).silu();
    final up = upProj.call(x);
    final inner = gate * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(inner);
    inner.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in gateProj.parameters().entries)
          'gate_proj.${e.key}': e.value,
        for (final e in downProj.parameters().entries)
          'down_proj.${e.key}': e.value,
        for (final e in upProj.parameters().entries)
          'up_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gateProj.loadWeights(_scoped(weights, 'gate_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
    upProj.loadWeights(_scoped(weights, 'up_proj'));
  }

  @override
  void dispose() {
    gateProj.dispose();
    downProj.dispose();
    upProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — per-head QK-norm, optional RoPE (local=true → RoPE, full=false → no RoPE)
//
// Post-norm style: the norm is applied AFTER the residual in the block.
// Block applies: h = x + postAttnNorm(attn(x))
// ---------------------------------------------------------------------------

final class _Exaone4Attention extends Module {
  _Exaone4Attention(MLXContext ctx, Exaone4Config cfg, bool isLocal)
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
        qNorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
        kNorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _useRope = isLocal || cfg.slidingWindowPattern == null,
        _rope = _buildRope(ctx, cfg);

  static RopeLayer _buildRope(MLXContext ctx, Exaone4Config cfg) {
    final scaling = cfg.ropeScaling;
    double ropeScale = 1.0;
    if (scaling != null && scaling['type'] == 'linear') {
      final factor = (scaling['factor'] as num?)?.toDouble();
      if (factor != null) ropeScale = 1.0 / factor;
    }
    return DefaultRope(
      dims: cfg.headDim,
      traditional: false,
      base: cfg.ropeTheta,
      scale: ropeScale,
    );
  }

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  RMSNorm qNorm;
  RMSNorm kNorm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final bool _useRope;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    final qFlat = qProj.call(x);
    // Per-head QK-norm: reshape to [B, L, H, Hd], norm, then transpose
    var q = qNorm.call(qFlat.reshape([b, s, numHeads, headDim]))
        .transpose([0, 2, 1, 3]);
    qFlat.dispose();

    final kFlat = kProj.call(x);
    var k = kNorm.call(kFlat.reshape([b, s, numKVHeads, headDim]))
        .transpose([0, 2, 1, 3]);
    kFlat.dispose();

    final v = vProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    if (_useRope) {
      final qRoped = _rope.call(q, offset);
      q.dispose();
      q = qRoped;
      final kRoped = _rope.call(k, offset);
      k.dispose();
      k = kRoped;
    }

    MLXArray? mask;
    if (s > 1) {
      mask = createCausalMask(ctx, n: s, offset: offset);
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
      maskMode: mask != null ? 'causal' : 'none',
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
  }
}

// ---------------------------------------------------------------------------
// Transformer block — post-norm architecture
//
//   r = attn(x)
//   h = x + postAttentionLayerNorm(r)
//   r = mlp(h)
//   out = h + postFeedforwardLayerNorm(r)
// ---------------------------------------------------------------------------

final class _Exaone4TransformerBlock extends Module {
  _Exaone4TransformerBlock(MLXContext ctx, Exaone4Config cfg, bool isLocal)
      : selfAttn = _Exaone4Attention(ctx, cfg, isLocal),
        mlp = _Exaone4MLP(ctx, cfg),
        postAttentionLayerNorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postFeedforwardLayerNorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Exaone4Attention selfAttn;
  _Exaone4MLP mlp;
  RMSNorm postAttentionLayerNorm;
  RMSNorm postFeedforwardLayerNorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final r = selfAttn.call(ctx, x, cache);
    final normedAttn = postAttentionLayerNorm.call(r);
    r.dispose();
    final h = x + normedAttn;
    normedAttn.dispose();

    final mlpOut = mlp.call(h);
    final normedMlp = postFeedforwardLayerNorm.call(mlpOut);
    mlpOut.dispose();
    final out = h + normedMlp;
    normedMlp.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in postAttentionLayerNorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
        for (final e in postFeedforwardLayerNorm.parameters().entries)
          'post_feedforward_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    postAttentionLayerNorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    postFeedforwardLayerNorm
        .loadWeights(_scoped(weights, 'post_feedforward_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    postAttentionLayerNorm.dispose();
    postFeedforwardLayerNorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model
// ---------------------------------------------------------------------------

final class _Exaone4InnerModel extends Module {
  _Exaone4InnerModel(MLXContext ctx, Exaone4Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers,
            (i) => _Exaone4TransformerBlock(ctx, cfg, cfg.isLocal(i))),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Exaone4TransformerBlock> layers;
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
// Public Exaone4Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// Exaone 4 language model.
///
/// Hybrid local/full attention architecture:
/// - `sliding_window_pattern` string (e.g. "LFLFL...") determines layer type;
///   'L' = local sliding-window, 'F' = full attention.
/// - Per-head QK-norm (RMSNorm on [B, L, H, Hd] before transpose).
/// - Post-norm residuals: `h = x + postAttnNorm(attn(x))`.
/// - Linear RoPE scaling via `rope_scaling.factor`.
/// - `RotatingKVCache` for local layers, `KVCacheSimple` for full layers.
///
/// Mirrors `Exaone4Model` from mlx-swift-lm.
final class Exaone4Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Exaone4Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Exaone4InnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Exaone4Config config;
  final _Exaone4InnerModel _model;
  final Linear? _lmHead;

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
    return List.generate(config.numHiddenLayers, (i) {
      if (config.isLocal(i)) {
        final sw = config.slidingWindow;
        if (sw != null) return RotatingKVCache(maxSize: sw);
      }
      return KVCacheSimple();
    });
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = <String, MLXArray>{...weights};
    if (config.tieWordEmbeddings) {
      result.remove('lm_head.weight');
    }
    return result;
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
