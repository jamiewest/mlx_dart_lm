import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class MiniCPMConfig {
  const MiniCPMConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.maxPositionEmbeddings = 4096,
    this.tieWordEmbeddings = true,
    this.scaleDepth = 1.4,
    this.scaleEmb = 1.0,
    this.dimModelBase = 256,
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

  /// Depth scaling factor — each residual is multiplied by
  /// `scaleDepth / sqrt(numHiddenLayers)`.
  final double scaleDepth;

  /// Embedding output is multiplied by this value.
  final double scaleEmb;

  /// Output logits are divided by `hiddenSize / dimModelBase` before lm_head.
  final int dimModelBase;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  /// Per-layer residual scale: `scaleDepth / sqrt(numHiddenLayers)`.
  double get residualScale => scaleDepth / math.sqrt(numHiddenLayers.toDouble());

  factory MiniCPMConfig.fromJson(Map<String, dynamic> j) => MiniCPMConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 4096,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? true,
        scaleDepth: (j['scale_depth'] as num?)?.toDouble() ?? 1.4,
        scaleEmb: (j['scale_emb'] as num?)?.toDouble() ?? 1.0,
        dimModelBase: j['dim_model_base'] as int? ?? 256,
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
// MLP — gate_proj · up_proj → silu → down_proj
// ---------------------------------------------------------------------------

final class _MiniCPMMLP extends Module {
  _MiniCPMMLP(MLXContext ctx, MiniCPMConfig cfg)
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

final class _MiniCPMAttention extends Module {
  _MiniCPMAttention(MLXContext ctx, MiniCPMConfig cfg)
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
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = initializeRope(
          ctx,
          dims: cfg.headDim,
          traditional: false,
          base: cfg.ropeTheta,
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
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    final q = qProj
        .call(x)
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final k = kProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final v = vProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

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
// Decoder layer — with residual scaling
// ---------------------------------------------------------------------------

final class _MiniCPMDecoderLayer extends Module {
  _MiniCPMDecoderLayer(MLXContext ctx, MiniCPMConfig cfg)
      : selfAttn = _MiniCPMAttention(ctx, cfg),
        mlp = _MiniCPMMLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _residualScale = cfg.residualScale;

  _MiniCPMAttention selfAttn;
  _MiniCPMMLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;
  final double _residualScale;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    // residual = x + attn * residualScale
    final scaledAttn = attnOut * MLXArray.float_(ctx, _residualScale);
    attnOut.dispose();
    final afterAttn = x + scaledAttn;
    scaledAttn.dispose();

    final normed2 = postAttentionLayernorm.call(afterAttn);
    final mlpOut = mlp.call(normed2);
    normed2.dispose();
    // residual = afterAttn + mlp * residualScale
    final scaledMlp = mlpOut * MLXArray.float_(ctx, _residualScale);
    mlpOut.dispose();
    final out = afterAttn + scaledMlp;
    scaledMlp.dispose();
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

final class _MiniCPMInnerModel extends Module {
  _MiniCPMInnerModel(MLXContext ctx, MiniCPMConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _MiniCPMDecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _scaleEmb = cfg.scaleEmb;

  Embedding embedTokens;
  List<_MiniCPMDecoderLayer> layers;
  RMSNorm norm;
  final double _scaleEmb;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
    if (_scaleEmb != 1.0) {
      final scaled = h * MLXArray.float_(ctx, _scaleEmb);
      h.dispose();
      h = scaled;
    }
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
// Public MiniCPMModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// MiniCPM language model.
///
/// Key differences from a standard LLaMA-style model:
/// - Each residual connection is scaled by `scaleDepth / sqrt(numLayers)`.
/// - Embedding output is multiplied by `scaleEmb`.
/// - Logits are divided by `hiddenSize / dimModelBase` before the lm_head.
///
/// Mirrors `MiniCPMModel` from mlx-swift-lm.
final class MiniCPMModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  MiniCPMModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _MiniCPMInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false),
        _outputScale = config.hiddenSize / config.dimModelBase;

  final MLXContext _ctx;
  final MiniCPMConfig config;
  final _MiniCPMInnerModel _model;
  final Linear? _lmHead;
  final double _outputScale;

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
    var h = _model.call(_ctx, input.tokens, cache);

    // Divide hidden states by outputScale before projecting to vocab
    if (_outputScale != 1.0) {
      final divisor = MLXArray.float_(_ctx, _outputScale);
      final scaled = h / divisor;
      divisor.dispose();
      h.dispose();
      h = scaled;
    }

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
    if (config.tieWordEmbeddings) {
      return {
        for (final e in weights.entries)
          if (e.key != 'lm_head.weight') e.key: e.value,
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
