import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class LFM2Config {
  const LFM2Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.vocabSize,
    required this.fullAttnIdxs,
    this.normEps = 1e-5,
    this.ropeTheta = 1000000.0,
    this.convBias = false,
    this.convLCache = 3,
    required this.blockDim,
    required this.blockFFDim,
    this.blockMultipleOf = 256,
    this.blockFFNDimMultiplier = 1.0,
    this.blockAutoAdjustFFDim = true,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int vocabSize;
  final List<int> fullAttnIdxs;
  final double normEps;
  final double ropeTheta;
  final bool convBias;
  final int convLCache;
  final int blockDim;
  final int blockFFDim;
  final int blockMultipleOf;
  final double blockFFNDimMultiplier;
  final bool blockAutoAdjustFFDim;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  /// FFN dim with optional auto-adjustment (mirrors Swift/Python).
  int get adjustedFFDim {
    if (!blockAutoAdjustFFDim) return blockFFDim;
    var d = (2 * blockFFDim) ~/ 3;
    d = (blockFFNDimMultiplier * d).round();
    d = blockMultipleOf * ((d + blockMultipleOf - 1) ~/ blockMultipleOf);
    return d;
  }

  bool isAttentionLayer(int i) => fullAttnIdxs.contains(i);

  factory LFM2Config.fromJson(Map<String, dynamic> j) {
    final numLayers = j['num_hidden_layers'] as int;
    final hiddenSize = j['hidden_size'] as int;

    List<int> fullAttnIdxs;
    if (j['full_attn_idxs'] != null) {
      fullAttnIdxs = (j['full_attn_idxs'] as List).cast<int>();
    } else if (j['layer_types'] != null) {
      final layerTypes = (j['layer_types'] as List).cast<String>();
      fullAttnIdxs = [
        for (var i = 0; i < layerTypes.length; i++)
          if (layerTypes[i] == 'full_attention') i,
      ];
    } else {
      fullAttnIdxs = List.generate(numLayers, (i) => i);
    }

    final blockDimRaw = j['block_dim'] as int? ?? hiddenSize;
    final blockFFDimRaw = j['block_ff_dim'] as int? ?? hiddenSize;

    return LFM2Config(
      hiddenSize: hiddenSize,
      numHiddenLayers: numLayers,
      numAttentionHeads: j['num_attention_heads'] as int,
      numKeyValueHeads: j['num_key_value_heads'] as int? ??
          j['num_attention_heads'] as int,
      vocabSize: j['vocab_size'] as int? ?? 65536,
      fullAttnIdxs: fullAttnIdxs,
      normEps: (j['norm_eps'] as num?)?.toDouble() ?? 1e-5,
      ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
      convBias: j['conv_bias'] as bool? ?? false,
      convLCache: j['conv_L_cache'] as int? ?? 3,
      blockDim: blockDimRaw,
      blockFFDim: blockFFDimRaw,
      blockMultipleOf: j['block_multiple_of'] as int? ?? 256,
      blockFFNDimMultiplier:
          (j['block_ffn_dim_multiplier'] as num?)?.toDouble() ?? 1.0,
      blockAutoAdjustFFDim: j['block_auto_adjust_ff_dim'] as bool? ?? true,
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
// Attention — per-head QK-norm (RMSNorm on [B,L,H,Hd] before transpose)
//
// Weight keys: q_proj, k_proj, v_proj, out_proj, q_layernorm, k_layernorm
// ---------------------------------------------------------------------------

final class _LFM2Attention extends Module {
  _LFM2Attention(MLXContext ctx, LFM2Config cfg)
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
        outProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        qLayernorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.normEps),
        kLayernorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.normEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = DefaultRope(
          dims: cfg.headDim,
          traditional: false,
          base: cfg.ropeTheta,
        );

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear outProj;
  RMSNorm qLayernorm;
  RMSNorm kLayernorm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Per-head QK-norm: reshape to [B,L,H,Hd], norm, transpose
    final q = qLayernorm
        .call(qProj.call(x).reshape([b, s, numHeads, headDim]))
        .transpose([0, 2, 1, 3]);
    final k = kLayernorm
        .call(kProj.call(x).reshape([b, s, numKVHeads, headDim]))
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
    final out = outProj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in qProj.parameters().entries) 'q_proj.${e.key}': e.value,
        for (final e in kProj.parameters().entries) 'k_proj.${e.key}': e.value,
        for (final e in vProj.parameters().entries) 'v_proj.${e.key}': e.value,
        for (final e in outProj.parameters().entries)
          'out_proj.${e.key}': e.value,
        for (final e in qLayernorm.parameters().entries)
          'q_layernorm.${e.key}': e.value,
        for (final e in kLayernorm.parameters().entries)
          'k_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
    qLayernorm.loadWeights(_scoped(weights, 'q_layernorm'));
    kLayernorm.loadWeights(_scoped(weights, 'k_layernorm'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    outProj.dispose();
    qLayernorm.dispose();
    kLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// ShortConv — depthwise Conv1d SSM layer
//
// in_proj: [hidden, 3*hidden] → split into B, C, x
// conv:    depthwise conv1d on B*x, kernel=lCache, groups=hidden
// out_proj: [hidden, hidden]
//
// Conv weight: [hidden, kernelSize, 1] in MLX format.
// Cache: MambaCache[0] = conv state [B, lCache-1, hidden]
// ---------------------------------------------------------------------------

final class _LFM2ShortConv extends Module {
  _LFM2ShortConv(MLXContext ctx, LFM2Config cfg)
      : inProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: 3 * cfg.hiddenSize,
            bias: cfg.convBias),
        outProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.hiddenSize,
            bias: cfg.convBias),
        convWeight = MLXArray.zeros(ctx, [cfg.hiddenSize, cfg.convLCache, 1]),
        _hiddenSize = cfg.hiddenSize,
        _lCache = cfg.convLCache;

  Linear inProj;
  Linear outProj;
  MLXArray convWeight; // [hidden, kernelSize, 1]
  final int _hiddenSize;
  final int _lCache;

  MLXArray call(MLXContext ctx, MLXArray x, MambaCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    final bcx = inProj.call(x); // [B, s, 3*H]
    final bPart = bcx.slice(start: [0, 0, 0], stop: [b, s, _hiddenSize]);
    final cPart = bcx.slice(
        start: [0, 0, _hiddenSize], stop: [b, s, 2 * _hiddenSize]);
    final xPart = bcx.slice(
        start: [0, 0, 2 * _hiddenSize], stop: [b, s, 3 * _hiddenSize]);
    bcx.dispose();

    final bx = bPart * xPart; // [B, s, H]
    bPart.dispose();
    xPart.dispose();

    // Prepend conv state (or zeros) to bx
    final MLXArray state;
    if (cache?.convState != null) {
      state = cache!.convState!;
    } else {
      state = MLXArray.zeros(ctx, [b, _lCache - 1, _hiddenSize],
          dtype: bx.dtype);
    }
    final bxPadded = concatenate(ctx, [state, bx], axis: 1); // [B, lCache-1+s, H]
    bx.dispose();

    // Save last lCache-1 steps to conv state
    if (cache != null) {
      final newState = bxPadded.slice(
          start: [0, bxPadded.dim(1) - (_lCache - 1), 0],
          stop: [b, bxPadded.dim(1), _hiddenSize]);
      cache.convState?.dispose();
      cache.convState = newState;
    }

    // Depthwise conv1d: [B, lCache-1+s, H] → [B, s, H]
    final convOut = conv1d(ctx, bxPadded, convWeight, groups: _hiddenSize);
    bxPadded.dispose();

    final y = cPart * convOut; // [B, s, H]
    cPart.dispose();
    convOut.dispose();

    final out = outProj.call(y);
    y.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in inProj.parameters().entries)
          'in_proj.${e.key}': e.value,
        for (final e in outProj.parameters().entries)
          'out_proj.${e.key}': e.value,
        'conv.weight': convWeight,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    inProj.loadWeights(_scoped(weights, 'in_proj'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
    if (weights['conv.weight'] case final w?) {
      convWeight.dispose();
      convWeight = w;
    }
  }

  @override
  void dispose() {
    inProj.dispose();
    outProj.dispose();
    convWeight.dispose();
  }
}

// ---------------------------------------------------------------------------
// MLP — SiLU-gated, keys: w1, w2, w3
// ---------------------------------------------------------------------------

final class _LFM2MLP extends Module {
  _LFM2MLP(MLXContext ctx, int dim, int ffDim)
      : w1 = Linear(ctx, inFeatures: dim, outFeatures: ffDim, bias: false),
        w2 = Linear(ctx, inFeatures: ffDim, outFeatures: dim, bias: false),
        w3 = Linear(ctx, inFeatures: dim, outFeatures: ffDim, bias: false);

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
// Decoder layer — switches between attention and conv
//
// Keys: self_attn or conv (operator), feed_forward, operator_norm, ffn_norm
// ---------------------------------------------------------------------------

final class _LFM2DecoderLayer extends Module {
  _LFM2DecoderLayer(MLXContext ctx, LFM2Config cfg, int layerIdx)
      : isAttentionLayer = cfg.isAttentionLayer(layerIdx),
        attention = cfg.isAttentionLayer(layerIdx)
            ? _LFM2Attention(ctx, cfg)
            : null,
        conv = cfg.isAttentionLayer(layerIdx)
            ? null
            : _LFM2ShortConv(ctx, cfg),
        feedForward =
            _LFM2MLP(ctx, cfg.blockDim, cfg.adjustedFFDim),
        operatorNorm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEps),
        ffnNorm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEps);

  final bool isAttentionLayer;
  _LFM2Attention? attention;
  _LFM2ShortConv? conv;
  _LFM2MLP feedForward;
  RMSNorm operatorNorm;
  RMSNorm ffnNorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = operatorNorm.call(x);
    final MLXArray r;
    if (isAttentionLayer) {
      r = attention!.call(ctx, normed, cache);
    } else {
      r = conv!.call(ctx, normed, cache is MambaCache ? cache : null);
    }
    normed.dispose();
    final h = x + r;
    r.dispose();
    final ffOut = feedForward.call(ffnNorm.call(h));
    final out = h + ffOut;
    ffOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    if (attention != null) {
      for (final e in attention!.parameters().entries) {
        result['self_attn.${e.key}'] = e.value;
      }
    }
    if (conv != null) {
      for (final e in conv!.parameters().entries) {
        result['conv.${e.key}'] = e.value;
      }
    }
    for (final e in feedForward.parameters().entries) {
      result['feed_forward.${e.key}'] = e.value;
    }
    for (final e in operatorNorm.parameters().entries) {
      result['operator_norm.${e.key}'] = e.value;
    }
    for (final e in ffnNorm.parameters().entries) {
      result['ffn_norm.${e.key}'] = e.value;
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention?.loadWeights(_scoped(weights, 'self_attn'));
    conv?.loadWeights(_scoped(weights, 'conv'));
    feedForward.loadWeights(_scoped(weights, 'feed_forward'));
    operatorNorm.loadWeights(_scoped(weights, 'operator_norm'));
    ffnNorm.loadWeights(_scoped(weights, 'ffn_norm'));
  }

  @override
  void dispose() {
    attention?.dispose();
    conv?.dispose();
    feedForward.dispose();
    operatorNorm.dispose();
    ffnNorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model — embedding_norm applied to embeddings
// ---------------------------------------------------------------------------

final class _LFM2InnerModel extends Module {
  _LFM2InnerModel(MLXContext ctx, LFM2Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers,
            (i) => _LFM2DecoderLayer(ctx, cfg, i)),
        embeddingNorm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.normEps);

  Embedding embedTokens;
  List<_LFM2DecoderLayer> layers;
  RMSNorm embeddingNorm;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
    for (var i = 0; i < layers.length; i++) {
      final cache = (caches != null && i < caches.length) ? caches[i] : null;
      final next = layers[i].call(ctx, h, cache);
      h.dispose();
      h = next;
    }
    final normed = embeddingNorm.call(h);
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
    for (final e in embeddingNorm.parameters().entries) {
      result['embedding_norm.${e.key}'] = e.value;
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    embedTokens.loadWeights(_scoped(weights, 'embed_tokens'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'layers.$i'));
    }
    embeddingNorm.loadWeights(_scoped(weights, 'embedding_norm'));
  }

  @override
  void dispose() {
    embedTokens.dispose();
    for (final l in layers) {
      l.dispose();
    }
    embeddingNorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public LFM2Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// LFM2 language model.
///
/// Hybrid architecture alternating between full attention layers and
/// depthwise Conv1d SSM layers:
/// - Attention layers: per-head QK-norm, standard RoPE, `KVCacheSimple`.
/// - Conv layers: `in_proj → split B/C/x → conv(B*x) → C*conv_out → out_proj`,
///   `MambaCache` for the conv window state.
/// - Layer selection via `full_attn_idxs` or `layer_types` in config.
/// - Tied embeddings (always, no lm_head).
///
/// Sanitize transposes conv weights from PyTorch [C,1,K] → MLX [C,K,1].
///
/// Mirrors `LFM2Model` from mlx-swift-lm.
final class LFM2Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  LFM2Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _LFM2InnerModel(ctx, config);

  final MLXContext _ctx;
  final LFM2Config config;
  final _LFM2InnerModel _model;

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
    final logits = h.matmul(_model.embedTokens.weight.T);
    h.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    return List.generate(config.numHiddenLayers, (i) {
      return config.isAttentionLayer(i) ? KVCacheSimple() : MambaCache();
    });
  }

  @override
  List<int> get kvHeads => List.generate(config.numHiddenLayers, (i) {
        return config.isAttentionLayer(i) ? config.numKeyValueHeads : 0;
      });

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = <String, MLXArray>{};
    for (final e in weights.entries) {
      var param = e.value;
      // Conv weight: transpose from PyTorch [C, 1, K] → MLX [C, K, 1]
      if (e.key.contains('conv.weight')) {
        final shape = param.shape;
        if (shape.length == 3 && shape.last > shape[1]) {
          param = param.transpose([0, 2, 1]);
        }
      }
      result[e.key] = param;
    }
    return result;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries) 'model.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
  }

  @override
  void dispose() {
    _model.dispose();
  }
}
