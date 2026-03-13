import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Mistral3TextConfig {
  const Mistral3TextConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.headDimensions,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.maxPositionEmbeddings,
    this.ropeTheta = 10000.0,
    this.ropeParameters,
    this.tieWordEmbeddings = false,
    required this.layerTypes,
    this.slidingWindow,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int? headDimensions;
  final double rmsNormEps;
  final int vocabSize;
  final int? maxPositionEmbeddings;
  final double ropeTheta;

  /// Optional `rope_parameters` dict from config (overrides ropeTheta if present).
  /// May contain: `rope_theta`, `llama_4_scaling_beta`,
  /// `original_max_position_embeddings`, `type`, etc.
  final Map<String, dynamic>? ropeParameters;

  final bool tieWordEmbeddings;
  final List<String> layerTypes;
  final int? slidingWindow;

  int get headDim => headDimensions ?? (hiddenSize ~/ numAttentionHeads);

  factory Mistral3TextConfig.fromJson(Map<String, dynamic> j) {
    final numLayers = j['num_hidden_layers'] as int;
    List<String> layerTypes;
    if (j['layer_types'] != null) {
      layerTypes = (j['layer_types'] as List).cast<String>();
    } else {
      layerTypes = List.filled(numLayers, 'full_attention');
    }
    final ropeParameters = j['rope_parameters'] as Map<String, dynamic>?;
    final ropeTheta = (ropeParameters?['rope_theta'] as num?)?.toDouble() ??
        (j['rope_theta'] as num?)?.toDouble() ??
        10000.0;
    return Mistral3TextConfig(
      hiddenSize: j['hidden_size'] as int,
      numHiddenLayers: numLayers,
      intermediateSize: j['intermediate_size'] as int,
      numAttentionHeads: j['num_attention_heads'] as int,
      numKeyValueHeads:
          j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
      headDimensions: j['head_dim'] as int?,
      rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
      vocabSize: j['vocab_size'] as int,
      maxPositionEmbeddings: j['max_position_embeddings'] as int?,
      ropeTheta: ropeTheta,
      ropeParameters: ropeParameters,
      tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
      layerTypes: layerTypes,
      slidingWindow: j['sliding_window'] as int?,
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

/// Llama-4 style position-dependent attention scale.
/// Returns shape [s, 1] to broadcast over [B, H, S, D] queries.
MLXArray _llama4AttnScale(
    MLXContext ctx, int offset, int s, double beta, int maxPosEmbed) {
  final positions = MLXArray.arange(
          ctx, offset.toDouble(), (offset + s).toDouble(), 1.0,
          dtype: MLXDtype.float32)
      .reshape([s]);
  final ratio = positions / MLXArray.float_(ctx, maxPosEmbed.toDouble());
  positions.dispose();
  final logged = (ratio.floor() + MLXArray.float_(ctx, 1.0)).log();
  ratio.dispose();
  final scaled = MLXArray.float_(ctx, 1.0) + MLXArray.float_(ctx, beta) * logged;
  logged.dispose();
  return scaled.reshape([s, 1]);
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

final class _Mistral3MLP extends Module {
  _Mistral3MLP(MLXContext ctx, Mistral3TextConfig cfg)
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
// Attention
// ---------------------------------------------------------------------------

final class _Mistral3Attention extends Module {
  _Mistral3Attention(MLXContext ctx, Mistral3TextConfig cfg)
      : wq = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: false),
        wk = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        wv = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        wo = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = initializeRope(
          ctx,
          dims: cfg.headDim,
          traditional: false,
          base: cfg.ropeTheta,
          scalingConfig: cfg.ropeParameters,
          maxPositionEmbeddings:
              cfg.maxPositionEmbeddings ?? cfg.hiddenSize,
        );

  Linear wq;
  Linear wk;
  Linear wv;
  Linear wo;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  // [attnScale] is [s, 1], applied to queries before dot product.
  MLXArray call(
      MLXContext ctx, MLXArray x, MLXArray? attnScale, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = wq
        .call(x)
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final k = wk
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final v = wv
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    final qRoped = _rope.call(q, offset);
    q.dispose();
    q = qRoped;
    final kRoped = _rope.call(k, offset);
    k.dispose();

    // Apply optional position-based attention scale
    if (attnScale != null) {
      // attnScale: [s, 1] → broadcast to [1, 1, s, 1] → [B, H, S, D]
      final qs = q * attnScale.reshape([1, 1, s, 1]);
      q.dispose();
      q = qs;
    }

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
        for (final e in wq.parameters().entries) 'q_proj.${e.key}': e.value,
        for (final e in wk.parameters().entries) 'k_proj.${e.key}': e.value,
        for (final e in wv.parameters().entries) 'v_proj.${e.key}': e.value,
        for (final e in wo.parameters().entries) 'o_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    wq.loadWeights(_scoped(weights, 'q_proj'));
    wk.loadWeights(_scoped(weights, 'k_proj'));
    wv.loadWeights(_scoped(weights, 'v_proj'));
    wo.loadWeights(_scoped(weights, 'o_proj'));
  }

  @override
  void dispose() {
    wq.dispose();
    wk.dispose();
    wv.dispose();
    wo.dispose();
  }
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

final class _Mistral3TransformerBlock extends Module {
  _Mistral3TransformerBlock(MLXContext ctx, Mistral3TextConfig cfg,
      this.isSliding)
      : attention = _Mistral3Attention(ctx, cfg),
        mlp = _Mistral3MLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  final bool isSliding;
  _Mistral3Attention attention;
  _Mistral3MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(
      MLXContext ctx, MLXArray x, MLXArray? attnScale, KVCache? cache) {
    final r = attention.call(ctx, inputLayernorm.call(x), attnScale, cache);
    final h = x + r;
    r.dispose();
    final mlpOut = mlp.call(postAttentionLayernorm.call(h));
    final out = h + mlpOut;
    mlpOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attention.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    attention.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model
// ---------------------------------------------------------------------------

final class _Mistral3InnerModel extends Module {
  _Mistral3InnerModel(MLXContext ctx, Mistral3TextConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = [
          for (final lt in cfg.layerTypes)
            _Mistral3TransformerBlock(ctx, cfg, lt == 'sliding_attention'),
        ],
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Mistral3TransformerBlock> layers;
  RMSNorm norm;

  MLXArray call(
      MLXContext ctx, Mistral3TextConfig cfg, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
    final s = tokens.dim(0);
    final offset = caches?.firstOrNull?.offset ?? 0;

    // Compute position-based attention scale once for all full-attention layers
    MLXArray? attnScale;
    final rp = cfg.ropeParameters;
    if (rp != null) {
      final beta = (rp['llama_4_scaling_beta'] as num?)?.toDouble();
      final origMaxPos = rp['original_max_position_embeddings'] as int?;
      if (beta != null && origMaxPos != null) {
        attnScale = _llama4AttnScale(ctx, offset, s, beta, origMaxPos);
      }
    }

    for (var i = 0; i < layers.length; i++) {
      final cache = (caches != null && i < caches.length) ? caches[i] : null;
      // Pass attnScale only for full-attention layers
      final scale = layers[i].isSliding ? null : attnScale;
      final next = layers[i].call(ctx, h, scale, cache);
      h.dispose();
      h = next;
    }
    attnScale?.dispose();

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
// Public Mistral3TextModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Mistral 3 text language model.
///
/// Hybrid full/sliding-window attention (`layer_types` array).
/// Supports Llama-4 style position-based attention scaling when
/// `llama_4_scaling_beta` is present in `rope_parameters`.
///
/// Mirrors `Mistral3TextModel` from mlx-swift-lm.
final class Mistral3TextModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Mistral3TextModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Mistral3InnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Mistral3TextConfig config;
  final _Mistral3InnerModel _model;
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
    final h = _model.call(_ctx, config, input.tokens, cache);
    final logits = _lmHead != null
        ? _lmHead.call(h)
        : h.matmul(_model.embedTokens.weight.T);
    h.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    return [
      for (final layer in _model.layers)
        layer.isSliding && config.slidingWindow != null
            ? RotatingKVCache(maxSize: config.slidingWindow!)
            : KVCacheSimple(),
    ];
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) => {
        for (final e in weights.entries)
          if (!e.key.contains('self_attn.rotary_emb.inv_freq')) e.key: e.value,
      };

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
