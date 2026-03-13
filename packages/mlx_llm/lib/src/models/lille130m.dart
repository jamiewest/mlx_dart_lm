import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Lille130mConfig {
  const Lille130mConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.vocabSize,
    this.layerNormEps = 1e-5,
    this.ropeTheta = 10000.0,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int vocabSize;
  final double layerNormEps;
  final double ropeTheta;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  /// FFN hidden dim computed as: round(8/3 * hiddenSize / 256) * 256, min 256.
  int get ffnDim {
    final numerator = (8 * hiddenSize) ~/ 3;
    final rounded = (numerator / 256.0).round();
    return math.max(256 * rounded, 256);
  }

  factory Lille130mConfig.fromJson(Map<String, dynamic> j) => Lille130mConfig(
        hiddenSize: j['n_embd'] as int,
        numHiddenLayers: j['n_layer'] as int,
        numAttentionHeads: j['n_head'] as int,
        numKeyValueHeads: j['n_kv_heads'] as int? ?? j['n_head'] as int,
        vocabSize: j['vocab_size'] as int,
        layerNormEps: (j['layer_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
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
// Attention — fused QKV projection, per-layer norm before QKV
//
// Weight keys: qkv_proj, out_proj, norm
// ---------------------------------------------------------------------------

final class _Lille130mAttention extends Module {
  _Lille130mAttention(MLXContext ctx, Lille130mConfig cfg)
      : qkvProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures:
                (cfg.numAttentionHeads + 2 * cfg.numKeyValueHeads) * cfg.headDim,
            bias: false),
        outProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.layerNormEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = DefaultRope(
          dims: cfg.headDim,
          traditional: true,
          base: cfg.ropeTheta,
        );

  Linear qkvProj;
  Linear outProj;
  RMSNorm norm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Apply norm before QKV projection
    final normed = norm.call(x);
    final qkv = qkvProj.call(normed);
    normed.dispose();

    final qSize = numHeads * headDim;
    final kvSize = numKVHeads * headDim;
    var q = qkv.slice(start: [0, 0, 0], stop: [b, s, qSize]);
    var k = qkv.slice(start: [0, 0, qSize], stop: [b, s, qSize + kvSize]);
    var v = qkv.slice(start: [0, 0, qSize + kvSize], stop: [b, s, qSize + 2 * kvSize]);
    qkv.dispose();

    q = q.reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    k = k.reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    v = v.reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

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
        for (final e in norm.parameters().entries) 'norm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qkvProj.loadWeights(_scoped(weights, 'qkv_proj'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
    norm.loadWeights(_scoped(weights, 'norm'));
  }

  @override
  void dispose() {
    qkvProj.dispose();
    outProj.dispose();
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// MLP — inner norm before gate/up/down
// ---------------------------------------------------------------------------

final class _Lille130mMLP extends Module {
  _Lille130mMLP(MLXContext ctx, Lille130mConfig cfg)
      : norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.layerNormEps),
        gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.ffnDim,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.ffnDim,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: cfg.ffnDim,
            outFeatures: cfg.hiddenSize,
            bias: false);

  RMSNorm norm;
  Linear gateProj;
  Linear upProj;
  Linear downProj;

  MLXArray call(MLXArray x) {
    final h = norm.call(x);
    final gate = gateProj.call(h).silu();
    final up = upProj.call(h);
    h.dispose();
    final inner = gate * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(inner);
    inner.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in norm.parameters().entries) 'norm.${e.key}': e.value,
        for (final e in gateProj.parameters().entries)
          'gate_proj.${e.key}': e.value,
        for (final e in upProj.parameters().entries)
          'up_proj.${e.key}': e.value,
        for (final e in downProj.parameters().entries)
          'down_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    norm.loadWeights(_scoped(weights, 'norm'));
    gateProj.loadWeights(_scoped(weights, 'gate_proj'));
    upProj.loadWeights(_scoped(weights, 'up_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
  }

  @override
  void dispose() {
    norm.dispose();
    gateProj.dispose();
    upProj.dispose();
    downProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Transformer block — pre-norm inside attention and MLP modules
//
//   h   = x + attention(x)   (norm applied inside attention before qkv)
//   out = h + feed_forward(h) (norm applied inside feed_forward)
// ---------------------------------------------------------------------------

final class _Lille130mBlock extends Module {
  _Lille130mBlock(MLXContext ctx, Lille130mConfig cfg)
      : attention = _Lille130mAttention(ctx, cfg),
        feedForward = _Lille130mMLP(ctx, cfg);

  _Lille130mAttention attention;
  _Lille130mMLP feedForward;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final attnOut = attention.call(ctx, x, cache);
    final h = x + attnOut;
    attnOut.dispose();
    final ffOut = feedForward.call(h);
    final out = h + ffOut;
    ffOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attention.parameters().entries)
          'attention.${e.key}': e.value,
        for (final e in feedForward.parameters().entries)
          'feed_forward.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention.loadWeights(_scoped(weights, 'attention'));
    feedForward.loadWeights(_scoped(weights, 'feed_forward'));
  }

  @override
  void dispose() {
    attention.dispose();
    feedForward.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model — key: tok_embeddings, outer key: transformer
// ---------------------------------------------------------------------------

final class _Lille130mInnerModel extends Module {
  _Lille130mInnerModel(MLXContext ctx, Lille130mConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers, (_) => _Lille130mBlock(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.layerNormEps);

  Embedding embedTokens;
  List<_Lille130mBlock> layers;
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
    embedTokens.loadWeights(_scoped(weights, 'tok_embeddings'));
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
// Public Lille130mModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Lille 130M language model.
///
/// Compact GPT-style model with fused QKV projection, pre-norm inside each
/// sub-module, traditional RoPE, and tied embeddings.
///
/// Weight key differences from standard models:
/// - `tok_embeddings` instead of `embed_tokens`
/// - Outer key `transformer` wraps the inner model
/// - Attention: `qkv_proj`, `out_proj`, `norm` (pre-norm before QKV)
/// - MLP: `gate_proj`, `up_proj`, `down_proj`, `norm` (pre-norm before MLP)
///
/// Mirrors `Lille130mModel` from mlx-swift-lm.
final class Lille130mModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Lille130mModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _transformer = _Lille130mInnerModel(ctx, config);

  final MLXContext _ctx;
  final Lille130mConfig config;
  final _Lille130mInnerModel _transformer;

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
    // Always tied embeddings
    final logits = h.matmul(_transformer.embedTokens.weight.T);
    h.dispose();
    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    return List.generate(config.numHiddenLayers, (_) => KVCacheSimple());
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) => {
        for (final e in weights.entries)
          if (!e.key.contains('rotary_emb')) e.key: e.value,
      };

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _transformer.parameters().entries)
          'transformer.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _transformer.loadWeights(_scoped(weights, 'transformer'));
  }

  @override
  void dispose() {
    _transformer.dispose();
  }
}
