import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Olmo2Config {
  const Olmo2Config({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.headDimensions,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 10000.0,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.maxPositionEmbeddings,
    this.tieWordEmbeddings = true,
    this.attentionBias = false,
    this.mlpBias = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int? headDimensions;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool ropeTraditional;
  final Map<String, dynamic>? ropeScaling;
  final int? maxPositionEmbeddings;
  final bool tieWordEmbeddings;
  final bool attentionBias;
  final bool mlpBias;

  int get headDim => headDimensions ?? (hiddenSize ~/ numAttentionHeads);

  factory Olmo2Config.fromJson(Map<String, dynamic> j) => Olmo2Config(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        headDimensions: j['head_dim'] as int?,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeTraditional: j['rope_traditional'] as bool? ?? false,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        maxPositionEmbeddings: j['max_position_embeddings'] as int?,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? true,
        attentionBias: j['attention_bias'] as bool? ?? false,
        mlpBias: j['mlp_bias'] as bool? ?? false,
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

final class _Olmo2MLP extends Module {
  _Olmo2MLP(MLXContext ctx, Olmo2Config cfg)
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
// Attention — QK-norm on full projection output (same pattern as OlmoE)
// ---------------------------------------------------------------------------

final class _Olmo2Attention extends Module {
  _Olmo2Attention(MLXContext ctx, Olmo2Config cfg)
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
        qNorm = RMSNorm(ctx,
            dims: cfg.numAttentionHeads * cfg.headDim,
            eps: cfg.rmsNormEps),
        kNorm = RMSNorm(ctx,
            dims: cfg.numKeyValueHeads * cfg.headDim,
            eps: cfg.rmsNormEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = initializeRope(
          ctx,
          dims: cfg.headDim,
          traditional: cfg.ropeTraditional,
          base: cfg.ropeTheta,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings:
              cfg.maxPositionEmbeddings ?? cfg.hiddenSize,
        );

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
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // QK-norm on full projection, then per-head reshape
    final qNormed = qNorm.call(qProj.call(x));
    final q = qNormed
        .reshape([b, s, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    qNormed.dispose();

    final kNormed = kNorm.call(kProj.call(x));
    final k = kNormed
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    kNormed.dispose();

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
// Transformer block — POST-norm architecture
//
// Olmo2 applies norms AFTER each sublayer (not before):
//   h = x  + postAttnNorm(attn(x))
//   out = h + postFFNorm(mlp(h))
// ---------------------------------------------------------------------------

final class _Olmo2TransformerBlock extends Module {
  _Olmo2TransformerBlock(MLXContext ctx, Olmo2Config cfg)
      : selfAttn = _Olmo2Attention(ctx, cfg),
        mlp = _Olmo2MLP(ctx, cfg),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postFeedforwardLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Olmo2Attention selfAttn;
  _Olmo2MLP mlp;
  RMSNorm postAttentionLayernorm;
  RMSNorm postFeedforwardLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    // Post-norm: norm is applied to the sublayer OUTPUT before adding residual
    final attnOut = selfAttn.call(ctx, x, cache);
    final normedAttn = postAttentionLayernorm.call(attnOut);
    attnOut.dispose();
    final h = x + normedAttn;
    normedAttn.dispose();

    final mlpOut = mlp.call(h);
    final normedMlp = postFeedforwardLayernorm.call(mlpOut);
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
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
        for (final e in postFeedforwardLayernorm.parameters().entries)
          'post_feedforward_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    postFeedforwardLayernorm
        .loadWeights(_scoped(weights, 'post_feedforward_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    postAttentionLayernorm.dispose();
    postFeedforwardLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model
// ---------------------------------------------------------------------------

final class _Olmo2InnerModel extends Module {
  _Olmo2InnerModel(MLXContext ctx, Olmo2Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _Olmo2TransformerBlock(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Olmo2TransformerBlock> layers;
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
// Public Olmo2Model — implements LanguageModel
// ---------------------------------------------------------------------------

/// OLMo 2 language model.
///
/// Key difference from standard LLaMA-style: **post-norm architecture** —
/// norms are applied to sublayer outputs before adding the residual, not to
/// sublayer inputs. QK-norms operate on the full (non-per-head) projection.
///
/// Mirrors `Olmo2Model` from mlx-swift-lm.
final class Olmo2Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Olmo2Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Olmo2InnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Olmo2Config config;
  final _Olmo2InnerModel _model;
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
