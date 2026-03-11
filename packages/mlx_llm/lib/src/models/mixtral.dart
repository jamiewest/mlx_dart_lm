import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class MixtralConfig {
  const MixtralConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.numLocalExperts,
    required this.numExpertsPerToken,
    this.maxPositionEmbeddings = 4096,
    this.rmsNormEps = 1e-5,
    required this.vocabSize,
    this.ropeTheta = 1e6,
    this.ropeTraditional = false,
    this.ropeScaling,
    this.tieWordEmbeddings = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int numLocalExperts;
  final int numExpertsPerToken;
  final int maxPositionEmbeddings;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool ropeTraditional;
  final Map<String, dynamic>? ropeScaling;
  final bool tieWordEmbeddings;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  factory MixtralConfig.fromJson(Map<String, dynamic> j) => MixtralConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        numLocalExperts: j['num_local_experts'] as int,
        numExpertsPerToken: j['num_experts_per_tok'] as int,
        maxPositionEmbeddings: j['max_position_embeddings'] as int? ?? 4096,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1e6,
        ropeTraditional: j['rope_traditional'] as bool? ?? false,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
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

MLXArray _additiveCausalMask(
  MLXContext ctx, {
  required int n,
  required int offset,
  required MLXDtype dtype,
}) {
  final boolMask = createCausalMask(ctx, n: n, offset: offset);
  final zeros = MLXArray.zeros(ctx, [1], dtype: dtype);
  final negLarge = MLXArray.fromFloats(ctx, [-1e9]).astype(dtype);
  final result = where(ctx, boolMask, zeros, negLarge);
  boolMask.dispose();
  zeros.dispose();
  negLarge.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// SwitchGLU — stacked expert weights [E, h, d], full-compute then gather
// ---------------------------------------------------------------------------

final class _SwitchGLU extends Module {
  _SwitchGLU(
    MLXContext ctx, {
    required int inputDims,
    required int hiddenDims,
    required int numExperts,
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

    final innerT = innerAll.transpose([1, 0, 2]); // [E, B*L, h]
    innerAll.dispose();
    final dw = downProj.transpose([0, 2, 1]); // [E, h, d]
    final outT = innerT.matmul(dw); // [E, B*L, d]
    innerT.dispose();
    dw.dispose();
    final outFlat = outT.transpose([1, 0, 2]); // [B*L, E, d]
    outT.dispose();
    return outFlat.reshape([b, s, e, d]); // [b, s, E, d]
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
    final selected = outFlat2.take(flatInds, axis: 0); // [b*s*k, d]
    outFlat2.dispose();
    flatInds.dispose();

    final selectedBSKD = selected.reshape([b, s, k, d]);
    selected.dispose();
    final scoresExpanded = scores.expandDims(3); // [b, s, k, 1]
    final weighted = selectedBSKD * scoresExpanded;
    selectedBSKD.dispose();
    scoresExpanded.dispose();

    final out = weighted.sum(axis: 2); // [b, s, d]
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

final class _MixtralSparseMoEBlock extends Module {
  _MixtralSparseMoEBlock(MLXContext ctx, MixtralConfig cfg)
      : gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numLocalExperts,
            bias: false),
        switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.intermediateSize,
            numExperts: cfg.numLocalExperts),
        _topK = cfg.numExpertsPerToken,
        _numExperts = cfg.numLocalExperts;

  Linear gate;
  _SwitchGLU switchMlp;
  final int _topK;
  final int _numExperts;

  MLXArray call(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final e = _numExperts;
    final k = _topK;

    // Router: compute logits, select top-k by argsort(-logits)
    final logits = gate.call(x); // [b, s, E]
    final negLogits = logits * MLXArray.float_(ctx, -1.0);
    final sortedInds = negLogits.argsort(axis: -1);
    negLogits.dispose();
    final inds = sortedInds.slice(
        start: [0, 0, 0], stop: [b, s, k]); // [b, s, k]
    sortedInds.dispose();

    // Gather gate logits for top-k indices and softmax
    final bl = b * s;
    final logitsFlat = logits.reshape([bl, e]);
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

    final logitsFlat1d = logitsFlat.reshape([bl * e]);
    logitsFlat.dispose();
    logits.dispose();

    final topKLogits =
        logitsFlat1d.take(flatInds, axis: 0).reshape([b, s, k]);
    logitsFlat1d.dispose();
    flatInds.dispose();

    final scores = topKLogits.softmax(axis: -1); // [b, s, k]
    topKLogits.dispose();

    final out = switchMlp.call(ctx, x, inds, scores);
    inds.dispose();
    scores.dispose();
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
// Attention
// ---------------------------------------------------------------------------

final class _MixtralAttention extends Module {
  _MixtralAttention(MLXContext ctx, MixtralConfig cfg)
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

    var q =
        qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k =
        kProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v =
        vProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;
    q = rope.call(q, offset);
    k = rope.call(k, offset);

    MLXArray? mask;
    if (s > 1) {
      mask = _additiveCausalMask(ctx, n: s, offset: offset, dtype: q.dtype);
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
      maskMode: mask != null ? 'array' : 'none',
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
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _MixtralDecoderLayer extends Module {
  _MixtralDecoderLayer(MLXContext ctx, MixtralConfig cfg)
      : selfAttn = _MixtralAttention(ctx, cfg),
        blockSparseMoe = _MixtralSparseMoEBlock(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _MixtralAttention selfAttn;
  _MixtralSparseMoEBlock blockSparseMoe;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final h = x + attnOut;
    attnOut.dispose();

    final normed2 = postAttentionLayernorm.call(h);
    final moeOut = blockSparseMoe.call(ctx, normed2);
    normed2.dispose();
    final out = h + moeOut;
    moeOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in blockSparseMoe.parameters().entries)
          'block_sparse_moe.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    blockSparseMoe.loadWeights(_scoped(weights, 'block_sparse_moe'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    blockSparseMoe.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _MixtralInnerModel extends Module {
  _MixtralInnerModel(MLXContext ctx, MixtralConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _MixtralDecoderLayer(ctx, cfg)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_MixtralDecoderLayer> layers;
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
    result.addAll(
        norm.parameters().map((k, v) => MapEntry('norm.$k', v)));
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
// Public MixtralModel
// ---------------------------------------------------------------------------

final class MixtralModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  MixtralModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _MixtralInnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final MixtralConfig config;
  final _MixtralInnerModel _model;
  final Linear _lmHead;

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

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = Map<String, MLXArray>.from(weights);

    // When tied, copy embed_tokens weight to lm_head so it loads correctly
    if (config.tieWordEmbeddings &&
        !result.containsKey('lm_head.weight') &&
        result.containsKey('model.embed_tokens.weight')) {
      result['lm_head.weight'] = result['model.embed_tokens.weight']!;
    }

    // Stack per-expert weights: w1→gate_proj, w2→down_proj, w3→up_proj
    const probe = 'model.layers.0.block_sparse_moe.experts.0.w1.weight';
    if (!result.containsKey(probe)) return result;

    for (var l = 0; l < config.numHiddenLayers; l++) {
      final prefix = 'model.layers.$l.block_sparse_moe';
      for (final pair in [
        ('w1', 'gate_proj'),
        ('w2', 'down_proj'),
        ('w3', 'up_proj'),
      ]) {
        final src = pair.$1;
        final dst = pair.$2;
        for (final k in ['weight', 'scales', 'biases']) {
          if (!result.containsKey('$prefix.experts.0.$src.$k')) continue;
          final tensors = [
            for (var e = 0; e < config.numLocalExperts; e++)
              result.remove('$prefix.experts.$e.$src.$k')!,
          ];
          result['$prefix.switch_mlp.$dst.$k'] =
              stack(_ctx, tensors, axis: 0);
        }
      }
    }
    return result;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in _model.parameters().entries)
          'model.${e.key}': e.value,
        for (final e in _lmHead.parameters().entries)
          'lm_head.${e.key}': e.value,
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
