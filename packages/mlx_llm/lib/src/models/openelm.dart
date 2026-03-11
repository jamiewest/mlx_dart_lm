import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration helpers
// ---------------------------------------------------------------------------

/// Rounds [v] up to the nearest multiple of [divisor], with a [minValue] floor.
int _makeDivisible(double v, {int divisor = 8, double? minValue}) {
  final minVal = minValue ?? divisor.toDouble();
  var result = math.max(minVal,
      (((v + divisor / 2) / divisor).floorToDouble() * divisor));
  if (result < 0.9 * v) result += divisor;
  return result.toInt();
}

/// Computes number of heads given model dim and head dim.
int _computeHeads(int modelDim, int headDim) {
  assert(modelDim % headDim == 0, 'modelDim must be divisible by headDim');
  return modelDim ~/ headDim;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class OpenELMConfig {
  const OpenELMConfig({
    required this.modelDim,
    required this.numTransformerLayers,
    required this.vocabSize,
    required this.headDimensions,
    required this.ffnDimDivisor,
    required this.numQueryHeads,
    required this.kvHeads,
    required this.ffnMultipliers,
    this.normalizeQkProjections = true,
    this.shareInputOutputLayers = true,
    this.rmsNormEps = 1e-6,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = false,
  });

  final int modelDim;
  final int numTransformerLayers;
  final int vocabSize;
  final int headDimensions;
  final int ffnDimDivisor;

  /// Per-layer query head counts (length == numTransformerLayers).
  final List<int> numQueryHeads;

  /// Per-layer KV head counts (length == numTransformerLayers).
  final List<int> kvHeads;

  /// Per-layer FFN multipliers (length == numTransformerLayers).
  final List<double> ffnMultipliers;

  final bool normalizeQkProjections;
  final bool shareInputOutputLayers;
  final double rmsNormEps;
  final double ropeTheta;
  final bool ropeTraditional;

  factory OpenELMConfig.fromJson(Map<String, dynamic> j) {
    final numLayers = j['num_transformer_layers'] as int;
    final modelDim = j['model_dim'] as int;
    final headDim = j['head_dim'] as int;
    final ffnDimDivisor = j['ffn_dim_divisor'] as int? ?? 256;

    const numGqaGroups = 4;
    final qkvMult = (j['qkv_multipliers'] as List?)
            ?.map((v) => (v as num).toDouble())
            .toList() ??
        [0.5, 1.0];
    final ffnMult = (j['ffn_multipliers'] as List?)
            ?.map((v) => (v as num).toDouble())
            .toList() ??
        [0.5, 4.0];

    // Build per-layer multiplier lists via linear interpolation
    List<double> buildRange(double lo, double hi, int n) {
      if (n == 1) return [lo];
      return List.generate(
          n,
          (i) =>
              ((lo + (hi - lo) * i / (n - 1)) * 100).roundToDouble() / 100);
    }

    final qkvMultipliers = buildRange(qkvMult[0], qkvMult[1], numLayers);
    final ffnMultipliers = buildRange(ffnMult[0], ffnMult[1], numLayers);

    // Compute per-layer query dims → query head counts
    final numQueryHeads = qkvMultipliers.map((double m) {
      final qDim = _makeDivisible(modelDim * m,
          divisor: headDim * numGqaGroups);
      return _computeHeads(qDim, headDim);
    }).toList();

    final kvHeads = numQueryHeads.map((int h) => h ~/ numGqaGroups).toList();

    return OpenELMConfig(
      modelDim: modelDim,
      numTransformerLayers: numLayers,
      vocabSize: j['vocab_size'] as int,
      headDimensions: headDim,
      ffnDimDivisor: ffnDimDivisor,
      numQueryHeads: numQueryHeads,
      kvHeads: kvHeads,
      ffnMultipliers: ffnMultipliers,
      normalizeQkProjections:
          j['normalize_qk_projections'] as bool? ?? true,
      shareInputOutputLayers:
          j['share_input_output_layers'] as bool? ?? true,
      rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
      ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
      ropeTraditional: j['rope_traditional'] as bool? ?? false,
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
// FFN — fused gate+up projection split 50/50, SiLU gate
// ---------------------------------------------------------------------------

final class _OpenELMFeedForwardNetwork extends Module {
  _OpenELMFeedForwardNetwork(MLXContext ctx, int modelDim, int intermediateDim)
      : proj1 = Linear(ctx,
            inFeatures: modelDim,
            outFeatures: 2 * intermediateDim,
            bias: false),
        proj2 = Linear(ctx,
            inFeatures: intermediateDim,
            outFeatures: modelDim,
            bias: false);

  Linear proj1;
  Linear proj2;

  MLXArray call(MLXArray x) {
    final ab = proj1.call(x);
    final parts = ab.split(2, axis: 2); // [gate, up]
    ab.dispose();
    final gate = parts[0].silu();
    final inner = gate * parts[1];
    gate.dispose();
    parts[0].dispose();
    parts[1].dispose();
    final out = proj2.call(inner);
    inner.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in proj1.parameters().entries) 'proj_1.${e.key}': e.value,
        for (final e in proj2.parameters().entries) 'proj_2.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    proj1.loadWeights(_scoped(weights, 'proj_1'));
    proj2.loadWeights(_scoped(weights, 'proj_2'));
  }

  @override
  void dispose() {
    proj1.dispose();
    proj2.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — fused QKV, per-layer head counts, optional per-head QK-norm
// ---------------------------------------------------------------------------

final class _OpenELMAttention extends Module {
  _OpenELMAttention(
      MLXContext ctx, OpenELMConfig cfg, int layerIdx)
      : numHeads = cfg.numQueryHeads[layerIdx],
        numKVHeads = cfg.kvHeads[layerIdx],
        headDim = cfg.headDimensions,
        scale = 1.0 / math.sqrt(cfg.headDimensions.toDouble()),
        qkvProj = Linear(ctx,
            inFeatures: cfg.modelDim,
            outFeatures: (cfg.numQueryHeads[layerIdx] +
                    2 * cfg.kvHeads[layerIdx]) *
                cfg.headDimensions,
            bias: false),
        outProj = Linear(ctx,
            inFeatures: cfg.numQueryHeads[layerIdx] * cfg.headDimensions,
            outFeatures: cfg.modelDim,
            bias: false),
        qNorm = cfg.normalizeQkProjections
            ? RMSNorm(ctx, dims: cfg.headDimensions, eps: cfg.rmsNormEps)
            : null,
        kNorm = cfg.normalizeQkProjections
            ? RMSNorm(ctx, dims: cfg.headDimensions, eps: cfg.rmsNormEps)
            : null,
        _rope = DefaultRope(
          dims: cfg.headDimensions,
          traditional: cfg.ropeTraditional,
          base: cfg.ropeTheta,
        );

  Linear qkvProj;
  Linear outProj;
  final RMSNorm? qNorm;
  final RMSNorm? kNorm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Fused QKV: [B, L, (numHeads+2*numKVHeads)*headDim]
    // Reshape to [B, numHeads+2*numKVHeads, L, headDim] then slice
    final qkv = qkvProj
        .call(x)
        .reshape([b, s, numHeads + 2 * numKVHeads, headDim])
        .transpose([0, 2, 1, 3]); // [B, H+2KV, L, headDim]

    var q = qkv.slice(
        start: [0, 0, 0, 0], stop: [b, numHeads, s, headDim]);
    var k = qkv.slice(
        start: [0, numHeads, 0, 0],
        stop: [b, numHeads + numKVHeads, s, headDim]);
    final v = qkv.slice(
        start: [0, numHeads + numKVHeads, 0, 0],
        stop: [b, numHeads + 2 * numKVHeads, s, headDim]);
    qkv.dispose();

    // Optional per-head QK-norm (norm dims = headDim, applied per head)
    if (qNorm != null) {
      final qN = qNorm!.call(q);
      q.dispose();
      q = qN;
      final kN = kNorm!.call(k);
      k.dispose();
      k = kN;
    }

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
        for (final e in qkvProj.parameters().entries)
          'qkv_proj.${e.key}': e.value,
        for (final e in outProj.parameters().entries)
          'out_proj.${e.key}': e.value,
        if (qNorm != null)
          for (final e in qNorm!.parameters().entries)
            'q_norm.${e.key}': e.value,
        if (kNorm != null)
          for (final e in kNorm!.parameters().entries)
            'k_norm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qkvProj.loadWeights(_scoped(weights, 'qkv_proj'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
    qNorm?.loadWeights(_scoped(weights, 'q_norm'));
    kNorm?.loadWeights(_scoped(weights, 'k_norm'));
  }

  @override
  void dispose() {
    qkvProj.dispose();
    outProj.dispose();
    qNorm?.dispose();
    kNorm?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer — per-layer varying attention and FFN dims
// ---------------------------------------------------------------------------

final class _OpenELMDecoderLayer extends Module {
  _OpenELMDecoderLayer(MLXContext ctx, OpenELMConfig cfg, int layerIdx)
      : attn = _OpenELMAttention(ctx, cfg, layerIdx),
        ffn = _OpenELMFeedForwardNetwork(
          ctx,
          cfg.modelDim,
          _makeDivisible(cfg.ffnMultipliers[layerIdx] * cfg.modelDim,
              divisor: cfg.ffnDimDivisor),
        ),
        attnNorm = RMSNorm(ctx, dims: cfg.modelDim, eps: cfg.rmsNormEps),
        ffnNorm = RMSNorm(ctx, dims: cfg.modelDim, eps: cfg.rmsNormEps);

  _OpenELMAttention attn;
  _OpenELMFeedForwardNetwork ffn;
  RMSNorm attnNorm;
  RMSNorm ffnNorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = attnNorm.call(x);
    final attnOut = attn.call(ctx, normed, cache);
    normed.dispose();
    final h = x + attnOut;
    attnOut.dispose();

    final normed2 = ffnNorm.call(h);
    final ffnOut = ffn.call(normed2);
    normed2.dispose();
    final out = h + ffnOut;
    ffnOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attn.parameters().entries) 'attn.${e.key}': e.value,
        for (final e in ffn.parameters().entries) 'ffn.${e.key}': e.value,
        for (final e in attnNorm.parameters().entries)
          'attn_norm.${e.key}': e.value,
        for (final e in ffnNorm.parameters().entries)
          'ffn_norm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attn.loadWeights(_scoped(weights, 'attn'));
    ffn.loadWeights(_scoped(weights, 'ffn'));
    attnNorm.loadWeights(_scoped(weights, 'attn_norm'));
    ffnNorm.loadWeights(_scoped(weights, 'ffn_norm'));
  }

  @override
  void dispose() {
    attn.dispose();
    ffn.dispose();
    attnNorm.dispose();
    ffnNorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _OpenELMTransformer extends Module {
  _OpenELMTransformer(MLXContext ctx, OpenELMConfig cfg)
      : tokenEmbeddings =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.modelDim),
        layers = List.generate(
            cfg.numTransformerLayers,
            (i) => _OpenELMDecoderLayer(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.modelDim, eps: cfg.rmsNormEps);

  Embedding tokenEmbeddings;
  List<_OpenELMDecoderLayer> layers;
  RMSNorm norm;

  MLXArray call(MLXContext ctx, MLXArray tokens, List<KVCache>? caches) {
    var h = tokenEmbeddings.call(tokens);
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
      for (final e in tokenEmbeddings.parameters().entries)
        'token_embeddings.${e.key}': e.value,
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
    tokenEmbeddings.loadWeights(_scoped(weights, 'token_embeddings'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'layers.$i'));
    }
    norm.loadWeights(_scoped(weights, 'norm'));
  }

  @override
  void dispose() {
    tokenEmbeddings.dispose();
    for (final l in layers) {
      l.dispose();
    }
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public OpenELMModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// OpenELM language model.
///
/// Key unique features:
/// - Per-layer variable query/KV head counts derived from `qkv_multipliers`.
/// - Per-layer variable FFN intermediate dims derived from `ffn_multipliers`.
/// - Fused QKV projection with index-based unequal split.
/// - Optional per-head QK-norm (`normalize_qk_projections`).
/// - Weight key `token_embeddings` (not `embed_tokens`).
/// - `share_input_output_layers` controls tied embeddings.
///
/// Mirrors `OpenELMModel` from mlx-swift-lm.
final class OpenELMModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  OpenELMModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _transformer = _OpenELMTransformer(ctx, config),
        _lmHead = config.shareInputOutputLayers
            ? null
            : Linear(ctx,
                inFeatures: config.modelDim,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final OpenELMConfig config;
  final _OpenELMTransformer _transformer;
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
    final h = _transformer.call(_ctx, input.tokens, cache);
    final logits = _lmHead != null
        ? _lmHead.call(h)
        : h.matmul(_transformer.tokenEmbeddings.weight.T);
    h.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    final maxSize = parameters?.maxKVSize;
    return List.generate(config.numTransformerLayers, (_) {
      if (maxSize != null) return RotatingKVCache(maxSize: maxSize);
      return KVCacheSimple();
    });
  }

  /// Per-layer KV head counts (non-uniform across layers).
  @override
  List<int> get kvHeads => config.kvHeads;

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _transformer.parameters().entries)
          'transformer.${e.key}': e.value,
        if (_lmHead != null)
          for (final e in _lmHead.parameters().entries)
            'lm_head.${e.key}': e.value,
      };

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) =>
      weights;

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _transformer.loadWeights(_scoped(weights, 'transformer'));
    _lmHead?.loadWeights(_scoped(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _transformer.dispose();
    _lmHead?.dispose();
  }
}
