import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class OlmoEConfig {
  const OlmoEConfig({
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
    this.maxPositionEmbeddings = 4096,
    this.tieWordEmbeddings = true,
    this.attentionBias = false,
    this.mlpBias = false,
    required this.numExperts,
    required this.numExpertsPerToken,
    this.normTopkProb = false,
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
  final int numExperts;
  final int numExpertsPerToken;
  final bool normTopkProb;

  int get headDim => headDimensions ?? (hiddenSize ~/ numAttentionHeads);

  factory OlmoEConfig.fromJson(Map<String, dynamic> j) => OlmoEConfig(
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
        numExperts: j['num_experts'] as int,
        numExpertsPerToken: j['num_experts_per_tok'] as int,
        normTopkProb: j['norm_topk_prob'] as bool? ?? false,
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
// SwitchGLU — identical logic to Qwen3MoE; stacked [E, h, d] expert weights
// ---------------------------------------------------------------------------

final class _OlmoESwitchGLU extends Module {
  _OlmoESwitchGLU(
    MLXContext ctx, {
    required int inputDims,
    required int hiddenDims,
    required int numExperts,
    bool bias = false,
  })  : _inputDims = inputDims,
        _hiddenDims = hiddenDims,
        _numExperts = numExperts,
        gateProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        upProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        downProj = MLXArray.zeros(ctx, [numExperts, inputDims, hiddenDims]);

  final int _inputDims;
  final int _hiddenDims;
  final int _numExperts;

  MLXArray gateProj; // [E, h, d]
  MLXArray upProj;   // [E, h, d]
  MLXArray downProj; // [E, d, h]

  MLXArray _computeAll(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final d = _inputDims;
    final h = _hiddenDims;
    final e = _numExperts;
    final bl = b * s;

    final xFlat = x.reshape([bl, d]);

    final gw = gateProj.reshape([e * h, d]);
    final gateAll = xFlat.matmul(gw.T).reshape([bl, e, h]);
    gw.dispose();

    final uw = upProj.reshape([e * h, d]);
    final upAll = xFlat.matmul(uw.T).reshape([bl, e, h]);
    uw.dispose();
    xFlat.dispose();

    final innerAll = gateAll.silu() * upAll;
    gateAll.dispose();
    upAll.dispose();

    // [E, B*L, h] @ [E, h, d] → [E, B*L, d] → [B*L, E, d]
    final innerT = innerAll.transpose([1, 0, 2]);
    innerAll.dispose();
    final dw = downProj.transpose([0, 2, 1]);
    final outT = innerT.matmul(dw);
    innerT.dispose();
    dw.dispose();
    final outFlat = outT.transpose([1, 0, 2]);
    outT.dispose();
    return outFlat.reshape([b, s, e, d]);
  }

  MLXArray _gatherAndSum(
      MLXContext ctx, MLXArray expertOuts, MLXArray inds, MLXArray scores) {
    final b = expertOuts.dim(0);
    final s = expertOuts.dim(1);
    final e = _numExperts;
    final d = _inputDims;
    final k = inds.dim(2);

    final outFlat = expertOuts.reshape([b * s, e, d]);
    final indsFlat = inds.reshape([b * s, k]);

    final posRange = MLXArray.arange(ctx, 0.0, (b * s).toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([b * s, 1])
        .repeat(k, axis: 1);
    final eArr = MLXArray.int_(ctx, e);
    final offsets = posRange * eArr;
    posRange.dispose();
    eArr.dispose();

    final flatInds = (offsets + indsFlat).reshape([b * s * k]);
    offsets.dispose();
    indsFlat.dispose();

    final outFlat2 = outFlat.reshape([b * s * e, d]);
    outFlat.dispose();
    final selected = outFlat2.take(flatInds, axis: 0);
    outFlat2.dispose();
    flatInds.dispose();

    final selectedBSKD = selected.reshape([b, s, k, d]);
    selected.dispose();
    final scoresExpanded = scores.expandDims(3);
    final weighted = selectedBSKD * scoresExpanded;
    selectedBSKD.dispose();
    scoresExpanded.dispose();

    final out = weighted.sum(axis: 2);
    weighted.dispose();
    return out;
  }

  MLXArray call(
      MLXContext ctx, MLXArray x, MLXArray inds, MLXArray scores) {
    final allOuts = _computeAll(ctx, x);
    final out = _gatherAndSum(ctx, allOuts, inds, scores);
    allOuts.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        'gate_proj.weight': gateProj,
        'up_proj.weight': upProj,
        'down_proj.weight': downProj,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights.containsKey('gate_proj.weight')) {
      gateProj.dispose();
      gateProj = weights['gate_proj.weight']!;
    }
    if (weights.containsKey('up_proj.weight')) {
      upProj.dispose();
      upProj = weights['up_proj.weight']!;
    }
    if (weights.containsKey('down_proj.weight')) {
      downProj.dispose();
      downProj = weights['down_proj.weight']!;
    }
  }

  @override
  void dispose() {
    gateProj.dispose();
    upProj.dispose();
    downProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Sparse MoE block
// ---------------------------------------------------------------------------

final class _OlmoESparseMoEBlock extends Module {
  _OlmoESparseMoEBlock(MLXContext ctx, OlmoEConfig cfg)
      : gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numExperts,
            bias: false),
        switchMlp = _OlmoESwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.intermediateSize,
            numExperts: cfg.numExperts,
            bias: cfg.mlpBias),
        _topK = cfg.numExpertsPerToken,
        _normTopkProb = cfg.normTopkProb;

  Linear gate;
  _OlmoESwitchGLU switchMlp;
  final int _topK;
  final bool _normTopkProb;

  MLXArray call(MLXContext ctx, MLXArray x) {
    final logits = gate.call(x);
    final scores = logits.softmax(axis: -1);

    final negLogits = logits * MLXArray.float_(ctx, -1.0);
    logits.dispose();
    final sortedInds = negLogits.argsort(axis: -1);
    negLogits.dispose();

    final b = x.dim(0);
    final s = x.dim(1);
    final e = scores.dim(2);
    final k = _topK;
    final bl = b * s;

    final inds =
        sortedInds.slice(start: [0, 0, 0], stop: [b, s, k]);
    sortedInds.dispose();

    // Gather routing scores for selected experts
    final scoresFlat = scores.reshape([bl, e]);
    final indsFlat = inds.reshape([bl, k]);

    final posRange = MLXArray.arange(ctx, 0.0, bl.toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([bl, 1])
        .repeat(k, axis: 1);
    final eArr = MLXArray.int_(ctx, e);
    final offsets = posRange * eArr;
    posRange.dispose();
    eArr.dispose();

    final flatInds = (offsets + indsFlat).reshape([bl * k]);
    offsets.dispose();
    indsFlat.dispose();

    final scoresFlat2 = scoresFlat.reshape([bl * e]);
    scoresFlat.dispose();
    scores.dispose();

    var selectedScores =
        scoresFlat2.take(flatInds, axis: 0).reshape([b, s, k]);
    scoresFlat2.dispose();
    flatInds.dispose();

    if (_normTopkProb) {
      final sumScores =
          selectedScores.sum(axis: -1, keepdims: true);
      final norm = selectedScores / sumScores;
      sumScores.dispose();
      selectedScores.dispose();
      selectedScores = norm;
    }

    final out = switchMlp.call(ctx, x, inds, selectedScores);
    inds.dispose();
    selectedScores.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in gate.parameters().entries) 'gate.${e.key}': e.value,
        for (final e in switchMlp.parameters().entries)
          'switch_mlp.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gate.loadWeights(_scoped(weights, 'gate'));
    switchMlp.loadWeights(_scoped(weights, 'switch_mlp'));
  }

  @override
  void dispose() {
    gate.dispose();
    switchMlp.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — QK-norm on full projection output (before per-head reshape)
// ---------------------------------------------------------------------------

final class _OlmoEAttention extends Module {
  _OlmoEAttention(MLXContext ctx, OlmoEConfig cfg)
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
        // QK-norm dims are the full projection sizes (not per-head)
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

    // QK-norm applied BEFORE per-head reshape (full projection dims)
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
// Transformer block — every layer uses MoE
// ---------------------------------------------------------------------------

final class _OlmoETransformerBlock extends Module {
  _OlmoETransformerBlock(MLXContext ctx, OlmoEConfig cfg)
      : attention = _OlmoEAttention(ctx, cfg),
        mlp = _OlmoESparseMoEBlock(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _OlmoEAttention attention;
  _OlmoESparseMoEBlock mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = attention.call(ctx, normed, cache);
    normed.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    final normed2 = postAttentionLayernorm.call(afterAttn);
    final mlpOut = mlp.call(ctx, normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
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

final class _OlmoEInnerModel extends Module {
  _OlmoEInnerModel(MLXContext ctx, OlmoEConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _OlmoETransformerBlock(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_OlmoETransformerBlock> layers;
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
// Public OlmoEModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// OLMoE language model.
///
/// Differences from Qwen3MoE:
/// - QK-norms are applied to the full (non-per-head) projection output.
/// - Every layer uses the sparse MoE MLP (no dense-layer fallback).
/// - Supports `attention_bias` and `mlp_bias` config options.
///
/// Mirrors `OlmoEModel` from mlx-swift-lm.
final class OlmoEModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  OlmoEModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _OlmoEInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final OlmoEConfig config;
  final _OlmoEInnerModel _model;
  final Linear? _lmHead;

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

  // -------------------------------------------------------------------------
  // Weight loading
  // -------------------------------------------------------------------------

  /// Stack per-expert weights into `mlp.switch_mlp.*.weight` shape `[E, h, d]`.
  ///
  /// Also handles optional quantization suffixes (`scales`, `biases`).
  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = Map<String, MLXArray>.of(weights);

    if (!result
        .containsKey('model.layers.0.mlp.experts.0.up_proj.weight')) {
      return result;
    }

    for (var l = 0; l < config.numHiddenLayers; l++) {
      final prefix = 'model.layers.$l';
      for (final n in ['up_proj', 'down_proj', 'gate_proj']) {
        for (final suffix in ['weight', 'scales', 'biases']) {
          final key0 = '$prefix.mlp.experts.0.$n.$suffix';
          if (result.containsKey(key0)) {
            final toStack = <MLXArray>[];
            for (var ei = 0; ei < config.numExperts; ei++) {
              toStack.add(
                  result.remove('$prefix.mlp.experts.$ei.$n.$suffix')!);
            }
            result['$prefix.mlp.switch_mlp.$n.$suffix'] =
                stack(toStack.first.context, toStack, axis: 0);
          }
        }
      }
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
