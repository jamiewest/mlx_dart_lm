import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class SmolLM3Config {
  const SmolLM3Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    this.headDimensions,
    required this.numKeyValueHeads,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.maxPositionEmbeddings,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = false,
    this.tieWordEmbeddings = true,
    this.attentionBias = false,
    this.mlpBias = false,
    this.noRopeLayerInterval = 4,
    required this.noRopeLayers,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int? headDimensions;
  final int numKeyValueHeads;
  final double rmsNormEps;
  final int vocabSize;
  final int? maxPositionEmbeddings;
  final double ropeTheta;
  final bool ropeTraditional;
  final bool tieWordEmbeddings;
  final bool attentionBias;
  final bool mlpBias;
  final int noRopeLayerInterval;

  /// Per-layer flag: 1 = use RoPE, 0 = NoPE (identity).
  final List<int> noRopeLayers;

  int get headDim => headDimensions ?? (hiddenSize ~/ numAttentionHeads);

  factory SmolLM3Config.fromJson(Map<String, dynamic> j) {
    final numLayers = j['num_hidden_layers'] as int;
    final interval = j['no_rope_layer_interval'] as int? ?? 4;
    List<int> noRopeLayers;
    if (j['no_rope_layers'] != null) {
      noRopeLayers = (j['no_rope_layers'] as List).cast<int>();
    } else {
      // 1 = RoPE, 0 = NoPE every `interval`-th layer (1-indexed).
      noRopeLayers = List.generate(
          numLayers, (i) => (i + 1) % interval != 0 ? 1 : 0);
    }
    return SmolLM3Config(
      hiddenSize: j['hidden_size'] as int,
      numHiddenLayers: numLayers,
      intermediateSize: j['intermediate_size'] as int,
      numAttentionHeads: j['num_attention_heads'] as int,
      headDimensions: j['head_dim'] as int?,
      numKeyValueHeads: j['num_key_value_heads'] as int? ??
          j['num_attention_heads'] as int,
      rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
      vocabSize: j['vocab_size'] as int,
      maxPositionEmbeddings: j['max_position_embeddings'] as int?,
      ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
      ropeTraditional: j['rope_traditional'] as bool? ?? false,
      tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? true,
      attentionBias: j['attention_bias'] as bool? ?? false,
      mlpBias: j['mlp_bias'] as bool? ?? false,
      noRopeLayerInterval: interval,
      noRopeLayers: noRopeLayers,
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
// MLP
// ---------------------------------------------------------------------------

final class _SmolLM3MLP extends Module {
  _SmolLM3MLP(MLXContext ctx, SmolLM3Config cfg)
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: cfg.mlpBias),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: cfg.mlpBias),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: cfg.mlpBias);

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
// Attention — optional NoPE (no positional encoding on some layers)
// ---------------------------------------------------------------------------

final class _SmolLM3Attention extends Module {
  _SmolLM3Attention(MLXContext ctx, SmolLM3Config cfg, bool useRope)
      : qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: cfg.attentionBias),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: cfg.attentionBias),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: cfg.attentionBias),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: cfg.attentionBias),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = useRope
            ? DefaultRope(
                dims: cfg.headDim,
                traditional: cfg.ropeTraditional,
                base: cfg.ropeTheta,
              )
            : null;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer? _rope;

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
    if (_rope != null) {
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
// Transformer block
// ---------------------------------------------------------------------------

final class _SmolLM3TransformerBlock extends Module {
  _SmolLM3TransformerBlock(MLXContext ctx, SmolLM3Config cfg, bool useRope)
      : selfAttn = _SmolLM3Attention(ctx, cfg, useRope),
        mlp = _SmolLM3MLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _SmolLM3Attention selfAttn;
  _SmolLM3MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final r = selfAttn.call(ctx, inputLayernorm.call(x), cache);
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
// Inner model
// ---------------------------------------------------------------------------

final class _SmolLM3InnerModel extends Module {
  _SmolLM3InnerModel(MLXContext ctx, SmolLM3Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers,
            (i) => _SmolLM3TransformerBlock(
                ctx,
                cfg,
                i < cfg.noRopeLayers.length
                    ? cfg.noRopeLayers[i] != 0
                    : true)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_SmolLM3TransformerBlock> layers;
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
// Public SmolLM3Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// SmolLM 3 language model.
///
/// Standard LLaMA-style architecture with NoPE (No Positional Encoding) layers.
/// Per-layer `noRopeLayers` array (1 = RoPE, 0 = NoPE identity) controls which
/// layers apply rotary embeddings. Default: NoPE every 4th layer.
///
/// Mirrors `SmolLM3Model` from mlx-swift-lm.
final class SmolLM3Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  SmolLM3Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _SmolLM3InnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final SmolLM3Config config;
  final _SmolLM3InnerModel _model;
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
    final result = <String, MLXArray>{
      for (final e in weights.entries)
        if (!e.key.contains('self_attn.rotary_emb.inv_freq')) e.key: e.value,
    };
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
