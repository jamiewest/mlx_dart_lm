import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class NanoChatConfig {
  const NanoChatConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.vocabSize,
    required this.intermediateSize,
    this.maxPositionEmbeddings = 2048,
    this.ropeTheta = 10000.0,
    this.rmsNormEps = 1e-5,
    this.logitsSoftcap = 15.0,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int vocabSize;
  final int intermediateSize;
  final int maxPositionEmbeddings;
  final double ropeTheta;
  final double rmsNormEps;
  final double logitsSoftcap;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  factory NanoChatConfig.fromJson(Map<String, dynamic> j) => NanoChatConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads: j['num_key_value_heads'] as int? ??
            j['num_attention_heads'] as int,
        vocabSize: j['vocab_size'] as int,
        intermediateSize: j['intermediate_size'] as int,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 2048,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        logitsSoftcap:
            (j['logits_softcap'] as num?)?.toDouble() ?? 15.0,
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

/// Scale-free RMSNorm (no learnable weight): x / sqrt(mean(x²) + eps).
MLXArray _functionalRMSNorm(MLXContext ctx, MLXArray x, double eps) {
  final sq = x * x;
  final variance = sq.mean(axis: -1, keepdims: true);
  sq.dispose();
  final norm = x / (variance + MLXArray.float_(ctx, eps)).sqrt();
  variance.dispose();
  return norm;
}

// ---------------------------------------------------------------------------
// Attention — custom RoPE frequencies, per-head functional RMSNorm after RoPE
//
// Weight keys: c_q, c_k, c_v, c_proj
// RoPE: precomputed -exp(i * log(theta) / (halfDim)) frequencies
// QK-norm: functional (no weight), applied after RoPE
// ---------------------------------------------------------------------------

final class _NanoChatAttention extends Module {
  _NanoChatAttention(MLXContext ctx, NanoChatConfig cfg)
      : cQ = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: false),
        cK = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        cV = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: false),
        cProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = DefaultRope(
          dims: cfg.headDim,
          traditional: false,
          base: cfg.ropeTheta,
        );

  Linear cQ;
  Linear cK;
  Linear cV;
  Linear cProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, NanoChatConfig cfg, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = cQ
        .call(x)
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    var k = cK
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final v = cV
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    final qRoped = _rope.call(q, offset);
    q.dispose();
    q = qRoped;
    final kRoped = _rope.call(k, offset);
    k.dispose();
    k = kRoped;

    // Functional QK-norm (no learnable weight) applied after RoPE
    final qNormed = _functionalRMSNorm(ctx, q, cfg.rmsNormEps);
    q.dispose();
    final kNormed = _functionalRMSNorm(ctx, k, cfg.rmsNormEps);
    k.dispose();

    MLXArray? mask;
    if (s > 1) {
      mask = createCausalMask(ctx, n: s, offset: offset);
    }

    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(kNormed, v);
    } else {
      fullK = kNormed;
      fullV = v;
    }

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: qNormed,
      keys: fullK,
      values: fullV,
      scale: scale,
      maskMode: mask != null ? 'causal' : 'none',
      mask: mask,
    );
    mask?.dispose();
    qNormed.dispose();
    if (cache == null) {
      kNormed.dispose();
      v.dispose();
    }

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = cProj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in cQ.parameters().entries) 'c_q.${e.key}': e.value,
        for (final e in cK.parameters().entries) 'c_k.${e.key}': e.value,
        for (final e in cV.parameters().entries) 'c_v.${e.key}': e.value,
        for (final e in cProj.parameters().entries) 'c_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    cQ.loadWeights(_scoped(weights, 'c_q'));
    cK.loadWeights(_scoped(weights, 'c_k'));
    cV.loadWeights(_scoped(weights, 'c_v'));
    cProj.loadWeights(_scoped(weights, 'c_proj'));
  }

  @override
  void dispose() {
    cQ.dispose();
    cK.dispose();
    cV.dispose();
    cProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// MLP — ReLU² activation (relu(x)²), keys: c_fc, c_proj
// ---------------------------------------------------------------------------

final class _NanoChatMLP extends Module {
  _NanoChatMLP(MLXContext ctx, NanoChatConfig cfg)
      : cFc = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: false),
        cProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: false);

  Linear cFc;
  Linear cProj;

  MLXArray call(MLXArray x) {
    final activated = cFc.call(x).relu();
    final squared = activated * activated;
    activated.dispose();
    final out = cProj.call(squared);
    squared.dispose();
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
// Transformer block — functional RMSNorm (no learnable weight) for pre-norm
//
// Weight keys: attn, mlp
// ---------------------------------------------------------------------------

final class _NanoChatBlock extends Module {
  _NanoChatBlock(MLXContext ctx, NanoChatConfig cfg)
      : attention = _NanoChatAttention(ctx, cfg),
        mlp = _NanoChatMLP(ctx, cfg);

  _NanoChatAttention attention;
  _NanoChatMLP mlp;

  MLXArray call(MLXContext ctx, NanoChatConfig cfg, MLXArray x, KVCache? cache) {
    final normedX = _functionalRMSNorm(ctx, x, cfg.rmsNormEps);
    final attnOut = attention.call(ctx, cfg, normedX, cache);
    normedX.dispose();
    final h = x + attnOut;
    attnOut.dispose();

    final normedH = _functionalRMSNorm(ctx, h, cfg.rmsNormEps);
    final mlpOut = mlp.call(normedH);
    normedH.dispose();
    final out = h + mlpOut;
    mlpOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attention.parameters().entries)
          'attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention.loadWeights(_scoped(weights, 'attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
  }

  @override
  void dispose() {
    attention.dispose();
    mlp.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model — key: wte (embeddings), h (layers), outer key: transformer
//
// Initial embedding is normalized with functional RMSNorm.
// Final hidden state is also normalized before lm_head.
// ---------------------------------------------------------------------------

final class _NanoChatInnerModel extends Module {
  _NanoChatInnerModel(MLXContext ctx, NanoChatConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers, (_) => _NanoChatBlock(ctx, cfg));

  Embedding embedTokens;
  List<_NanoChatBlock> layers;

  MLXArray call(MLXContext ctx, NanoChatConfig cfg, MLXArray tokens, List<KVCache>? caches) {
    var h = embedTokens.call(tokens);
    // Normalize embedding output
    final normed0 = _functionalRMSNorm(ctx, h, cfg.rmsNormEps);
    h.dispose();
    h = normed0;

    for (var i = 0; i < layers.length; i++) {
      final cache = (caches != null && i < caches.length) ? caches[i] : null;
      final next = layers[i].call(ctx, cfg, h, cache);
      h.dispose();
      h = next;
    }

    // Final norm
    final out = _functionalRMSNorm(ctx, h, cfg.rmsNormEps);
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in embedTokens.parameters().entries)
        'wte.${e.key}': e.value,
    };
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['h.$i.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    embedTokens.loadWeights(_scoped(weights, 'wte'));
    for (var i = 0; i < layers.length; i++) {
      layers[i].loadWeights(_scoped(weights, 'h.$i'));
    }
  }

  @override
  void dispose() {
    embedTokens.dispose();
    for (final l in layers) {
      l.dispose();
    }
  }
}

// ---------------------------------------------------------------------------
// Public NanoChatModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// NanoChat language model.
///
/// Compact model with:
/// - Functional RMSNorm (no learnable weight) for pre-norm and embedding norm.
/// - ReLU² MLP activation (relu(x)²).
/// - Custom RoPE with functional per-head QK-norm (no weight) applied after RoPE.
/// - Logit soft-capping: `cap * tanh(logits / cap)`.
/// - Non-tied lm_head.
///
/// Weight key layout:
/// - `transformer.wte` — embeddings
/// - `transformer.h.{i}.attn.{c_q,c_k,c_v,c_proj}` — attention
/// - `transformer.h.{i}.mlp.{c_fc,c_proj}` — MLP
/// - `lm_head` — output projection
///
/// Mirrors `NanoChatModel` from mlx-swift-lm.
final class NanoChatModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  NanoChatModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _transformer = _NanoChatInnerModel(ctx, config),
        lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final NanoChatConfig config;
  final _NanoChatInnerModel _transformer;
  Linear lmHead;

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
    final h = _transformer.call(_ctx, config, input.tokens, cache);
    final rawLogits = lmHead.call(h);
    h.dispose();
    // Logit soft-capping: cap * tanh(logits / cap)
    final cap = config.logitsSoftcap;
    final capArr = MLXArray.float_(_ctx, cap);
    final logits = capArr * (rawLogits / capArr).tanh();
    rawLogits.dispose();
    capArr.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) =>
      List.generate(config.numHiddenLayers, (_) => KVCacheSimple());

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) =>
      weights;

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _transformer.parameters().entries)
          'transformer.${e.key}': e.value,
        for (final e in lmHead.parameters().entries)
          'lm_head.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _transformer.loadWeights(_scoped(weights, 'transformer'));
    lmHead.loadWeights(_scoped(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _transformer.dispose();
    lmHead.dispose();
  }
}
