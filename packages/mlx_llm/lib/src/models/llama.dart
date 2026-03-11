import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class LlamaConfig {
  const LlamaConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.headDimOverride,
    this.maxPositionEmbeddings = 4096,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 500000.0,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.tieWordEmbeddings = false,
    this.attentionBias = false,
    this.mlpBias = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int? headDimOverride;
  final int maxPositionEmbeddings;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool ropeTraditional;

  /// Optional `rope_scaling` map from `config.json`.
  ///
  /// Parsed as-is and forwarded to [initializeRope] to select the correct
  /// RoPE variant (e.g. `"llama3"`, `"yarn"`, etc.).
  final Map<String, dynamic>? ropeScaling;

  final bool tieWordEmbeddings;
  final bool attentionBias;
  final bool mlpBias;

  int get headDim => headDimOverride ?? hiddenSize ~/ numAttentionHeads;

  factory LlamaConfig.fromJson(Map<String, dynamic> j) => LlamaConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        headDimOverride: j['head_dim'] as int?,
        maxPositionEmbeddings: j['max_position_embeddings'] as int? ?? 4096,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 500000.0,
        ropeTraditional: j['rope_traditional'] as bool? ?? false,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        attentionBias: j['attention_bias'] as bool? ?? false,
        mlpBias: j['mlp_bias'] as bool? ?? false,
      );
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Filter [weights] to those whose keys start with [prefix], stripping it.
Map<String, MLXArray> _scoped(Map<String, MLXArray> weights, String prefix) {
  final p = '$prefix.';
  return {
    for (final e in weights.entries)
      if (e.key.startsWith(p)) e.key.substring(p.length): e.value,
  };
}

/// Build an additive causal mask — shape [n, offset+n] — from a boolean mask.
///
/// True (attend) → 0.0, False (masked) → -1e9.
MLXArray _additiveCausalMask(
  MLXContext ctx, {
  required int n,
  required int offset,
  required MLXDtype dtype,
}) {
  final boolMask = createCausalMask(ctx, n: n, offset: offset);
  final zeros = MLXArray.zeros(ctx, [1], dtype: dtype);
  final negLarge = MLXArray.fromFloats(ctx, [-1e9])
      .astype(dtype);
  final result = where(ctx, boolMask, zeros, negLarge);
  boolMask.dispose();
  zeros.dispose();
  negLarge.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

final class _LlamaMLP extends Module {
  _LlamaMLP(MLXContext ctx, LlamaConfig cfg)
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: cfg.mlpBias),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.intermediateSize,
            bias: cfg.mlpBias),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: cfg.mlpBias);

  Linear gateProj;
  Linear upProj;
  Linear downProj;

  MLXArray call(MLXArray x) {
    final gate = gateProj.call(x).silu();
    final up = upProj.call(x);
    final gated = gate * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(gated);
    gated.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...gateProj.parameters().map((k, v) => MapEntry('gate_proj.$k', v)),
        ...upProj.parameters().map((k, v) => MapEntry('up_proj.$k', v)),
        ...downProj.parameters().map((k, v) => MapEntry('down_proj.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gateProj.loadWeights(_scoped(weights, 'gate_proj'));
    upProj.loadWeights(_scoped(weights, 'up_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
  }

  @override
  void dispose() {
    gateProj.dispose();
    upProj.dispose();
    downProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

final class _LlamaAttention extends Module {
  _LlamaAttention(MLXContext ctx, LlamaConfig cfg)
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
        rope = initializeRope(
          ctx,
          dims: cfg.headDim,
          base: cfg.ropeTheta,
          traditional: cfg.ropeTraditional,
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
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Project.
    var q = qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k = kProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = vProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

    // RoPE (offset = tokens already in cache before this step).
    final offset = cache?.offset ?? 0;
    q = rope.call(q, offset);
    k = rope.call(k, offset);

    // Build attention mask before updating the cache (uses pre-update offset).
    MLXArray? mask;
    if (s > 1) {
      mask = _additiveCausalMask(ctx, n: s, offset: offset, dtype: q.dtype);
    }

    // Update KV cache.
    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(k, v);
    } else {
      fullK = k;
      fullV = v;
    }

    // Scaled dot-product attention.
    // MLX fast attention supports GQA natively (numHeads may differ from numKVHeads).
    final attnOut = scaledDotProductAttention(
      ctx,
      queries: q,
      keys: fullK,
      values: fullV,
      scale: scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    mask?.dispose();
    q.dispose();
    if (cache == null) {
      k.dispose();
      v.dispose();
    }

    // Reshape back and project output.
    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = oProj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...qProj.parameters().map((k, v) => MapEntry('q_proj.$k', v)),
        ...kProj.parameters().map((k, v) => MapEntry('k_proj.$k', v)),
        ...vProj.parameters().map((k, v) => MapEntry('v_proj.$k', v)),
        ...oProj.parameters().map((k, v) => MapEntry('o_proj.$k', v)),
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
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _LlamaDecoderLayer extends Module {
  _LlamaDecoderLayer(MLXContext ctx, LlamaConfig cfg)
      : selfAttn = _LlamaAttention(ctx, cfg),
        mlp = _LlamaMLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _LlamaAttention selfAttn;
  _LlamaMLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    // Self-attention with pre-norm and residual.
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    // MLP with pre-norm and residual.
    final normed2 = postAttentionLayernorm.call(afterAttn);
    final mlpOut = mlp.call(normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...selfAttn.parameters().map((k, v) => MapEntry('self_attn.$k', v)),
        ...mlp.parameters().map((k, v) => MapEntry('mlp.$k', v)),
        ...inputLayernorm.parameters().map((k, v) => MapEntry('input_layernorm.$k', v)),
        ...postAttentionLayernorm.parameters().map((k, v) => MapEntry('post_attention_layernorm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm.loadWeights(_scoped(weights, 'post_attention_layernorm'));
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

final class _LlamaInnerModel extends Module {
  _LlamaInnerModel(MLXContext ctx, LlamaConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers,
            (_) => _LlamaDecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_LlamaDecoderLayer> layers;
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
      ...embedTokens.parameters().map((k, v) => MapEntry('embed_tokens.$k', v)),
    };
    for (var i = 0; i < layers.length; i++) {
      for (final e in layers[i].parameters().entries) {
        result['layers.$i.${e.key}'] = e.value;
      }
    }
    result.addAll(norm.parameters().map((k, v) => MapEntry('norm.$k', v)));
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
// Public LlamaModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Llama-family language model.
///
/// Mirrors `LlamaModel` from mlx-swift-lm. Implements [LanguageModel] and
/// [KVCacheDimensionProvider] so [newCache] is handled automatically.
final class LlamaModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  LlamaModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _LlamaInnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final LlamaConfig config;
  final _LlamaInnerModel _model;
  final Linear _lmHead;

  // -------------------------------------------------------------------------
  // LanguageModel
  // -------------------------------------------------------------------------

  @override
  PrepareResult prepare(LMInput input, List<KVCache> cache,
      {int? windowSize}) {
    final tokens = input.text.tokens; // shape [seqLen]
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
    final logits = _lmHead.call(h);
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
    // If lm_head is missing and embeddings are tied, share the weight.
    if (!weights.containsKey('lm_head.weight') &&
        weights.containsKey('model.embed_tokens.weight')) {
      return {
        ...weights,
        'lm_head.weight': weights['model.embed_tokens.weight']!,
      };
    }
    return weights;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ..._model.parameters().map((k, v) => MapEntry('model.$k', v)),
        ..._lmHead.parameters().map((k, v) => MapEntry('lm_head.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _model.loadWeights(_scoped(weights, 'model'));
    _lmHead.loadWeights(_scoped(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _model.dispose();
    _lmHead.dispose();
  }
}
