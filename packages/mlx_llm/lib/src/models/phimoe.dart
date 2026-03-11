import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class PhiMoEConfig {
  const PhiMoEConfig({
    this.vocabSize = 32064,
    this.hiddenSize = 4096,
    this.intermediateSize = 6400,
    this.numHiddenLayers = 32,
    this.numAttentionHeads = 32,
    this.numKeyValueHeads = 8,
    this.maxPositionEmbeddings = 131072,
    this.originalMaxPositionEmbeddings = 4096,
    this.rmsNormEps = 1e-6,
    this.ropeScaling,
    this.numLocalExperts = 16,
    this.numExpertsPerToken = 2,
    this.ropeTheta = 10000.0,
  });

  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int maxPositionEmbeddings;
  final int originalMaxPositionEmbeddings;
  final double rmsNormEps;
  final Map<String, dynamic>? ropeScaling;
  final int numLocalExperts;
  final int numExpertsPerToken;
  final double ropeTheta;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  factory PhiMoEConfig.fromJson(Map<String, dynamic> j) => PhiMoEConfig(
        vocabSize: j['vocab_size'] as int? ?? 32064,
        hiddenSize: j['hidden_size'] as int? ?? 4096,
        intermediateSize: j['intermediate_size'] as int? ?? 6400,
        numHiddenLayers: j['num_hidden_layers'] as int? ?? 32,
        numAttentionHeads: j['num_attention_heads'] as int? ?? 32,
        numKeyValueHeads: j['num_key_value_heads'] as int? ?? 8,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 131072,
        originalMaxPositionEmbeddings:
            j['original_max_position_embeddings'] as int? ?? 4096,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        numLocalExperts: j['num_local_experts'] as int? ?? 16,
        numExpertsPerToken: j['num_experts_per_tok'] as int? ?? 2,
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
// SwitchGLU — stacked [E, h, d] expert weights (same as Qwen3MoE / OlmoE)
// ---------------------------------------------------------------------------

final class _PhiMoESwitchGLU extends Module {
  _PhiMoESwitchGLU(
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
// Sparse MoE block — keyed `block_sparse_moe`
// ---------------------------------------------------------------------------

final class _PhiMoESparseMoeBlock extends Module {
  _PhiMoESparseMoeBlock(MLXContext ctx, PhiMoEConfig cfg)
      : gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numLocalExperts,
            bias: false),
        switchMlp = _PhiMoESwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.intermediateSize,
            numExperts: cfg.numLocalExperts),
        _topK = cfg.numExpertsPerToken;

  Linear gate;
  _PhiMoESwitchGLU switchMlp;
  final int _topK;

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

    final scoresFlat = scores.reshape([bl, e]);
    scores.dispose();
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

    final selectedScores =
        scoresFlat2.take(flatInds, axis: 0).reshape([b, s, k]);
    scoresFlat2.dispose();
    flatInds.dispose();

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
// Attention — biased projections, SuScaledRoPE via initializeRope
// ---------------------------------------------------------------------------

final class _PhiMoEAttention extends Module {
  _PhiMoEAttention(MLXContext ctx, PhiMoEConfig cfg)
      : qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim,
            bias: true),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: true),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim,
            bias: true),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: true),
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
// Decoder layer — LayerNorm (not RMSNorm), MoE keyed `block_sparse_moe`
// ---------------------------------------------------------------------------

final class _PhiMoEDecoderLayer extends Module {
  _PhiMoEDecoderLayer(MLXContext ctx, PhiMoEConfig cfg)
      : selfAttn = _PhiMoEAttention(ctx, cfg),
        blockSparseMoe = _PhiMoESparseMoeBlock(ctx, cfg),
        inputLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _PhiMoEAttention selfAttn;
  _PhiMoESparseMoeBlock blockSparseMoe;
  LayerNorm inputLayernorm;
  LayerNorm postAttentionLayernorm;

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
// Inner model
// ---------------------------------------------------------------------------

final class _PhiMoEInnerModel extends Module {
  _PhiMoEInnerModel(MLXContext ctx, PhiMoEConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (_) => _PhiMoEDecoderLayer(ctx, cfg)),
        norm = LayerNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_PhiMoEDecoderLayer> layers;
  LayerNorm norm;

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
// Public PhiMoEModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// PhiMoE language model.
///
/// Differences from Qwen3MoE:
/// - Uses `LayerNorm` (not `RMSNorm`) for all norms.
/// - All attention projections have `bias: true`; `lm_head` also has bias.
/// - MoE block is keyed `block_sparse_moe` (not `mlp`).
/// - Uses `SuScaledRoPE` via `initializeRope` with `longrope`/`su` scaling.
///
/// Mirrors `PhiMoEModel` from mlx-swift-lm.
final class PhiMoEModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  PhiMoEModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _PhiMoEInnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: true);

  final MLXContext _ctx;
  final PhiMoEConfig config;
  final _PhiMoEInnerModel _model;
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

  /// Stack per-expert weights from `block_sparse_moe.experts.N.*` into
  /// `block_sparse_moe.switch_mlp.*` shape `[E, h, d]`.
  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = Map<String, MLXArray>.of(weights);

    if (!result.containsKey(
        'model.layers.0.block_sparse_moe.experts.0.w1.weight')) {
      return result;
    }

    for (var l = 0; l < config.numHiddenLayers; l++) {
      final prefix = 'model.layers.$l.block_sparse_moe';
      // PhiMoE uses w1/w2/w3 keys in Python but switch_mlp uses gate/down/up
      for (final (src, dst) in [
        ('w1', 'gate_proj'),
        ('w2', 'down_proj'),
        ('w3', 'up_proj'),
      ]) {
        for (final suffix in ['weight', 'scales', 'biases']) {
          final key0 = '$prefix.experts.0.$src.$suffix';
          if (result.containsKey(key0)) {
            final toStack = <MLXArray>[];
            for (var ei = 0; ei < config.numLocalExperts; ei++) {
              toStack.add(
                  result.remove('$prefix.experts.$ei.$src.$suffix')!);
            }
            result['$prefix.switch_mlp.$dst.$suffix'] =
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
