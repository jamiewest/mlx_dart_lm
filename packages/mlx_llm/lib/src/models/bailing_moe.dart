import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class BailingMoeConfig {
  const BailingMoeConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.moeIntermediateSize,
    required this.numExperts,
    required this.numSharedExperts,
    required this.numExpertsPerToken,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.numHiddenLayers,
    required this.vocabSize,
    required this.firstKDenseReplace,
    this.rmsNormEps = 1e-6,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.useBias = false,
    this.useQKVBias = false,
    this.useQKNorm = false,
    this.tieWordEmbeddings = false,
    this.partialRotaryFactor = 1.0,
    this.moeRouterEnableExpertBias = false,
    this.routedScalingFactor = 1.0,
    this.scoreFunction = 'softmax',
    this.nGroup = 1,
    this.topkGroup = 4,
    this.normTopkProb = false,
    this.maxPositionEmbeddings,
    this.moeSharedExpertIntermediateSize,
  });

  final int hiddenSize;
  final int intermediateSize;
  final int moeIntermediateSize;
  final int numExperts;
  final int numSharedExperts;
  final int numExpertsPerToken;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int numHiddenLayers;
  final int vocabSize;
  final int firstKDenseReplace;
  final double rmsNormEps;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final bool useBias;
  final bool useQKVBias;
  final bool useQKNorm;
  final bool tieWordEmbeddings;
  final double partialRotaryFactor;
  final bool moeRouterEnableExpertBias;
  final double routedScalingFactor;
  final String scoreFunction;
  final int nGroup;
  final int topkGroup;
  final bool normTopkProb;
  final int? maxPositionEmbeddings;
  final int? moeSharedExpertIntermediateSize;

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get ropeDim => (headDim * partialRotaryFactor).round();

  int get sharedExpertDim =>
      (moeSharedExpertIntermediateSize ?? moeIntermediateSize) *
      numSharedExperts;

  factory BailingMoeConfig.fromJson(Map<String, dynamic> j) =>
      BailingMoeConfig(
        hiddenSize: j['hidden_size'] as int,
        intermediateSize: j['intermediate_size'] as int,
        moeIntermediateSize: j['moe_intermediate_size'] as int,
        numExperts: j['num_experts'] as int,
        numSharedExperts: j['num_shared_experts'] as int? ?? 0,
        numExpertsPerToken: j['num_experts_per_tok'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads: j['num_key_value_heads'] as int? ??
            j['num_attention_heads'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        vocabSize: j['vocab_size'] as int,
        firstKDenseReplace: j['first_k_dense_replace'] as int? ?? 0,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        useBias: j['use_bias'] as bool? ?? false,
        useQKVBias: j['use_qkv_bias'] as bool? ?? false,
        useQKNorm: j['use_qk_norm'] as bool? ?? false,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 1.0,
        moeRouterEnableExpertBias:
            j['moe_router_enable_expert_bias'] as bool? ?? false,
        routedScalingFactor:
            (j['routed_scaling_factor'] as num?)?.toDouble() ?? 1.0,
        scoreFunction: j['score_function'] as String? ?? 'softmax',
        nGroup: j['n_group'] as int? ?? 1,
        topkGroup: j['topk_group'] as int? ?? 4,
        normTopkProb: j['norm_topk_prob'] as bool? ?? false,
        maxPositionEmbeddings: j['max_position_embeddings'] as int?,
        moeSharedExpertIntermediateSize:
            j['moe_shared_expert_intermediate_size'] as int?,
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
// SwitchGLU — stacked expert weights [E, h, d], full-compute then gather
// ---------------------------------------------------------------------------

final class _SwitchGLU extends Module {
  _SwitchGLU(
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
  MLXArray upProj; // [E, h, d]
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
    if (weights['gate_proj.weight'] case final w?) {
      gateProj.dispose();
      gateProj = w;
    }
    if (weights['up_proj.weight'] case final w?) {
      upProj.dispose();
      upProj = w;
    }
    if (weights['down_proj.weight'] case final w?) {
      downProj.dispose();
      downProj = w;
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
// Attention — fused QKV (`query_key_value`), optional QK-norm, partial RoPE
// ---------------------------------------------------------------------------

final class _BailingMoeAttention extends Module {
  _BailingMoeAttention(MLXContext ctx, BailingMoeConfig cfg)
      : qkv = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures:
                (cfg.numAttentionHeads + 2 * cfg.numKeyValueHeads) * cfg.headDim,
            bias: cfg.useQKVBias),
        wo = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize,
            bias: cfg.useBias),
        qNorm = cfg.useQKNorm
            ? RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps)
            : null,
        kNorm = cfg.useQKNorm
            ? RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps)
            : null,
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        _rope = DefaultRope(
          dims: cfg.ropeDim,
          traditional: false,
          base: cfg.ropeTheta,
        );

  Linear qkv;
  Linear wo;
  RMSNorm? qNorm;
  RMSNorm? kNorm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    final qSize = numHeads * headDim;
    final kvSize = numKVHeads * headDim;
    final qkvOut = qkv.call(x);
    final queries0 = qkvOut.slice(start: [0, 0, 0], stop: [b, s, qSize]);
    final keys0 = qkvOut.slice(start: [0, 0, qSize], stop: [b, s, qSize + kvSize]);
    final values0 = qkvOut.slice(
        start: [0, 0, qSize + kvSize], stop: [b, s, qSize + 2 * kvSize]);
    qkvOut.dispose();

    // Reshape to [B, L, H, Hd], optional QK-norm, then transpose
    var queries = queries0.reshape([b, s, numHeads, headDim]);
    var keys = keys0.reshape([b, s, numKVHeads, headDim]);
    queries0.dispose();
    keys0.dispose();

    if (qNorm != null) queries = qNorm!.call(queries);
    if (kNorm != null) keys = kNorm!.call(keys);

    queries = queries.transpose([0, 2, 1, 3]);
    keys = keys.transpose([0, 2, 1, 3]);
    final values = values0
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    values0.dispose();

    final offset = cache?.offset ?? 0;
    final qRoped = _rope.call(queries, offset);
    queries.dispose();
    final kRoped = _rope.call(keys, offset);
    keys.dispose();

    MLXArray? mask;
    if (s > 1) {
      mask = createCausalMask(ctx, n: s, offset: offset);
    }

    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(kRoped, values);
    } else {
      fullK = kRoped;
      fullV = values;
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
      values.dispose();
    }

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = wo.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in qkv.parameters().entries)
        'query_key_value.${e.key}': e.value,
      for (final e in wo.parameters().entries) 'dense.${e.key}': e.value,
    };
    if (qNorm != null) {
      for (final e in qNorm!.parameters().entries) {
        result['query_layernorm.${e.key}'] = e.value;
      }
    }
    if (kNorm != null) {
      for (final e in kNorm!.parameters().entries) {
        result['key_layernorm.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qkv.loadWeights(_scoped(weights, 'query_key_value'));
    wo.loadWeights(_scoped(weights, 'dense'));
    qNorm?.loadWeights(_scoped(weights, 'query_layernorm'));
    kNorm?.loadWeights(_scoped(weights, 'key_layernorm'));
  }

  @override
  void dispose() {
    qkv.dispose();
    wo.dispose();
    qNorm?.dispose();
    kNorm?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Dense MLP
// ---------------------------------------------------------------------------

final class _BailingMoeDenseMLP extends Module {
  _BailingMoeDenseMLP(MLXContext ctx, BailingMoeConfig cfg, {int? hiddenDims})
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: hiddenDims ?? cfg.intermediateSize,
            bias: cfg.useBias),
        downProj = Linear(ctx,
            inFeatures: hiddenDims ?? cfg.intermediateSize,
            outFeatures: cfg.hiddenSize,
            bias: cfg.useBias),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: hiddenDims ?? cfg.intermediateSize,
            bias: cfg.useBias);

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
// MoE block — group-based gate + SwitchGLU + optional shared experts
//
// Gate (`gate_proj`): sigmoid scores + expert bias; select top-topkGroup groups
// by sum of top-2 per group, then top-k among eligible experts.
// ---------------------------------------------------------------------------

final class _BailingMoeSparseMoeBlock extends Module {
  _BailingMoeSparseMoeBlock(MLXContext ctx, BailingMoeConfig cfg)
      : switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.numExperts,
            bias: cfg.useBias),
        gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numExperts,
            bias: false),
        expertBias = cfg.moeRouterEnableExpertBias
            ? MLXArray.zeros(ctx, [cfg.numExperts])
            : null,
        sharedExperts = cfg.numSharedExperts > 0
            ? _BailingMoeDenseMLP(ctx, cfg, hiddenDims: cfg.sharedExpertDim)
            : null,
        _topK = cfg.numExpertsPerToken,
        _nGroup = cfg.nGroup,
        _topkGroup = cfg.topkGroup,
        _numExperts = cfg.numExperts,
        _routedScalingFactor = cfg.routedScalingFactor,
        _normTopkProb = cfg.normTopkProb,
        _scoreFunction = cfg.scoreFunction;

  _SwitchGLU switchMlp;
  Linear gate;
  MLXArray? expertBias;
  _BailingMoeDenseMLP? sharedExperts;
  final int _topK;
  final int _nGroup;
  final int _topkGroup;
  final int _numExperts;
  final double _routedScalingFactor;
  final bool _normTopkProb;
  final String _scoreFunction;

  (MLXArray, MLXArray) _groupSelect(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);

    final logits = gate.call(x); // [b, s, E]
    final scores = _scoreFunction == 'sigmoid'
        ? logits.sigmoid().astype(MLXDtype.float32)
        : logits.softmax(axis: -1).astype(MLXDtype.float32);

    // Add expert bias for group selection
    final MLXArray scoresForChoice;
    if (expertBias != null) {
      scoresForChoice = scores + expertBias!;
    } else {
      scoresForChoice = scores;
    }

    // Group scores: [b, s, nGroup, E/nGroup]
    final groupScores =
        scoresForChoice.reshape([b, s, _nGroup, _numExperts ~/ _nGroup]);

    // Top-2 per group, sum → [b, s, nGroup]
    final top2 = groupScores.sort(axis: -1);
    final lastTwo = top2.slice(
        start: [0, 0, 0, (_numExperts ~/ _nGroup) - 2],
        stop: [b, s, _nGroup, _numExperts ~/ _nGroup]);
    final groupTopSum = lastTwo.sum(axis: -1, keepdims: true); // [b,s,nGroup,1]
    top2.dispose();
    lastTwo.dispose();
    groupScores.dispose();
    if (expertBias != null) scoresForChoice.dispose();

    // Select topkGroup groups: argsort descending, take first topkGroup
    final negGroupSum = groupTopSum * MLXArray.float_(ctx, -1.0);
    groupTopSum.dispose();
    final groupSorted = negGroupSum.argsort(axis: -2); // [b,s,nGroup,1]
    negGroupSum.dispose();
    // Keep only first topkGroup groups
    final selectedGroups =
        groupSorted.slice(start: [0, 0, 0, 0], stop: [b, s, _topkGroup, 1]);
    groupSorted.dispose();

    // Build mask: zero out scores for non-selected groups
    // Zero out entire groups not selected — build a zeroed copy
    // Simplification: use argsort on flat scores among selected experts
    // For now: gather eligible experts via mask trick
    // We set scores for unselected groups to 0 by creating a binary mask.
    // Flatten group indices and expert scores to select top-k.

    // Convert selected group indices [b,s,topkGroup,1] → [b,s,topkGroup] flat
    final selGroupFlat =
        selectedGroups.reshape([b, s, _topkGroup]); // group idx
    selectedGroups.dispose();

    // Expert per group
    final epg = _numExperts ~/ _nGroup;

    // Build flat expert indices for selected groups: selGroup * epg + [0..epg)
    // Shape: [b, s, topkGroup, epg] → [b, s, topkGroup*epg]
    final groupRange = MLXArray.arange(ctx, 0.0, epg.toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([1, 1, 1, epg])
        .repeat(b, axis: 0)
        .repeat(s, axis: 1)
        .repeat(_topkGroup, axis: 2);
    final selGroupExp = selGroupFlat
        .reshape([b, s, _topkGroup, 1])
        .repeat(epg, axis: 3);
    selGroupFlat.dispose();
    final epgArr = MLXArray.int_(ctx, epg);
    final eligibleInds =
        (selGroupExp * epgArr + groupRange).reshape([b, s, _topkGroup * epg]);
    selGroupExp.dispose();
    groupRange.dispose();
    epgArr.dispose();

    // Gather scores for eligible experts
    final scoresFlat = scores.reshape([b * s, _numExperts]);
    final eligFlat = eligibleInds.reshape([b * s, _topkGroup * epg]);
    final posRange2 = MLXArray.arange(ctx, 0.0, (b * s).toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([b * s, 1])
        .repeat(_topkGroup * epg, axis: 1);
    final eArr2 = MLXArray.int_(ctx, _numExperts);
    final offsets2 = posRange2 * eArr2;
    posRange2.dispose();
    eArr2.dispose();
    final flatEligInds = (offsets2 + eligFlat).reshape([b * s * _topkGroup * epg]);
    offsets2.dispose();
    eligFlat.dispose();
    final scoresFlat2 = scoresFlat.reshape([b * s * _numExperts]);
    scoresFlat.dispose();
    final eligScores =
        scoresFlat2.take(flatEligInds, axis: 0).reshape([b, s, _topkGroup * epg]);
    scoresFlat2.dispose();
    flatEligInds.dispose();
    scores.dispose();
    logits.dispose();

    // Top-k among eligible
    final negEligScores = eligScores * MLXArray.float_(ctx, -1.0);
    final eligSorted = negEligScores.argsort(axis: -1);
    negEligScores.dispose();
    final topKLocalInds =
        eligSorted.slice(start: [0, 0, 0], stop: [b, s, _topK]);
    eligSorted.dispose();

    // Gather expert indices from eligibleInds
    final eligIndFlat2 = eligibleInds.reshape([b * s, _topkGroup * epg]);
    eligibleInds.dispose();
    final topKLocalFlat = topKLocalInds.reshape([b * s, _topK]);
    final posRange3 = MLXArray.arange(ctx, 0.0, (b * s).toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([b * s, 1])
        .repeat(_topK, axis: 1);
    final epgKArr = MLXArray.int_(ctx, _topkGroup * epg);
    final offsets3 = posRange3 * epgKArr;
    posRange3.dispose();
    epgKArr.dispose();
    final flatTopKInds =
        (offsets3 + topKLocalFlat).reshape([b * s * _topK]);
    offsets3.dispose();
    topKLocalFlat.dispose();
    final eligIndFlat3 = eligIndFlat2.reshape([b * s * _topkGroup * epg]);
    eligIndFlat2.dispose();
    final expertInds =
        eligIndFlat3.take(flatTopKInds, axis: 0).reshape([b, s, _topK]);
    eligIndFlat3.dispose();
    flatTopKInds.dispose();

    // Gather scores for top-k
    final eligScoresFlat = eligScores.reshape([b * s, _topkGroup * epg]);
    eligScores.dispose();
    final posRange4 = MLXArray.arange(ctx, 0.0, (b * s).toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([b * s, 1])
        .repeat(_topK, axis: 1);
    final epgKArr2 = MLXArray.int_(ctx, _topkGroup * epg);
    final offsets4 = posRange4 * epgKArr2;
    posRange4.dispose();
    epgKArr2.dispose();
    final topKLocalIndsFlat2 = topKLocalInds.reshape([b * s * _topK]);
    topKLocalInds.dispose();
    final flatTopKScoreInds =
        (offsets4 + topKLocalIndsFlat2).reshape([b * s * _topK]);
    offsets4.dispose();
    topKLocalIndsFlat2.dispose();
    final eligScoresFlat2 = eligScoresFlat.reshape([b * s * _topkGroup * epg]);
    eligScoresFlat.dispose();
    var expertScores = eligScoresFlat2
        .take(flatTopKScoreInds, axis: 0)
        .reshape([b, s, _topK]);
    eligScoresFlat2.dispose();
    flatTopKScoreInds.dispose();

    if (_normTopkProb) {
      final sumScores = expertScores.sum(axis: -1, keepdims: true);
      final norm = expertScores / sumScores;
      sumScores.dispose();
      expertScores.dispose();
      expertScores = norm;
    }

    final scaled = expertScores *
        MLXArray.float_(ctx, _routedScalingFactor);
    expertScores.dispose();

    return (expertInds, scaled);
  }

  MLXArray call(MLXContext ctx, MLXArray x) {
    final (inds, scores) = _groupSelect(ctx, x);
    var out = switchMlp.call(ctx, x, inds, scores);
    inds.dispose();
    scores.dispose();
    if (sharedExperts != null) {
      final shared = sharedExperts!.call(x);
      final combined = out + shared;
      out.dispose();
      shared.dispose();
      out = combined;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in switchMlp.parameters().entries)
        'switch_mlp.${e.key}': e.value,
      for (final e in gate.parameters().entries) 'gate.gate_proj.${e.key}': e.value,
    };
    if (expertBias != null) result['gate.expert_bias'] = expertBias!;
    if (sharedExperts != null) {
      for (final e in sharedExperts!.parameters().entries) {
        result['shared_experts.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    switchMlp.loadWeights(_scoped(weights, 'switch_mlp'));
    gate.loadWeights(_scoped(weights, 'gate.gate_proj'));
    if (expertBias != null && weights['gate.expert_bias'] != null) {
      expertBias!.dispose();
      expertBias = weights['gate.expert_bias'];
    }
    sharedExperts?.loadWeights(_scoped(weights, 'shared_experts'));
  }

  @override
  void dispose() {
    switchMlp.dispose();
    gate.dispose();
    expertBias?.dispose();
    sharedExperts?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Transformer block — pre-norm, dense MLP for first k layers, MoE otherwise
// ---------------------------------------------------------------------------

final class _BailingMoeTransformerBlock extends Module {
  _BailingMoeTransformerBlock(MLXContext ctx, BailingMoeConfig cfg, int layerIdx)
      : attention = _BailingMoeAttention(ctx, cfg),
        mlp = (cfg.numExperts > 0 && layerIdx >= cfg.firstKDenseReplace)
            ? _BailingMoeSparseMoeBlock(ctx, cfg)
            : _BailingMoeDenseMLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _BailingMoeAttention attention;
  Module mlp; // _BailingMoeSparseMoeBlock or _BailingMoeDenseMLP
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final r = attention.call(ctx, inputLayernorm.call(x), cache);
    final h = x + r;
    r.dispose();
    final MLXArray mlpOut;
    if (mlp is _BailingMoeSparseMoeBlock) {
      mlpOut = (mlp as _BailingMoeSparseMoeBlock)
          .call(ctx, postAttentionLayernorm.call(h));
    } else {
      mlpOut = (mlp as _BailingMoeDenseMLP).call(postAttentionLayernorm.call(h));
    }
    final out = h + mlpOut;
    mlpOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in attention.parameters().entries)
          'attention.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    attention.loadWeights(_scoped(weights, 'attention'));
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
// Inner model — embedding key: word_embeddings
// ---------------------------------------------------------------------------

final class _BailingMoeInnerModel extends Module {
  _BailingMoeInnerModel(MLXContext ctx, BailingMoeConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers,
            (i) => _BailingMoeTransformerBlock(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_BailingMoeTransformerBlock> layers;
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
        'word_embeddings.${e.key}': e.value,
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
    embedTokens.loadWeights(_scoped(weights, 'word_embeddings'));
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
// Public BailingMoeModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Bailing MoE language model (Ling-family models).
///
/// Hybrid dense/MoE architecture:
/// - Fused `query_key_value` projection with optional per-head QK-norm.
/// - Partial RoPE (`partial_rotary_factor`).
/// - First `first_k_dense_replace` layers use dense MLP; rest use MoE.
/// - Group-based gate routing: sigmoid scores, top-2 per group, select
///   top topkGroup groups, then top-k among eligible experts.
/// - Optional shared experts added to MoE output.
///
/// Weight key differences: `word_embeddings` for embeddings,
/// `query_key_value`/`dense` for attention, `gate.gate_proj`/`gate.expert_bias`
/// for router. Expert weights stacked in sanitize.
///
/// Mirrors `BailingMoeModel` from mlx-swift-lm.
final class BailingMoeModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  BailingMoeModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _BailingMoeInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final BailingMoeConfig config;
  final _BailingMoeInnerModel _model;
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
  List<KVCache> newCache(GenerateParameters? parameters) =>
      List.generate(config.numHiddenLayers, (_) => KVCacheSimple());

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    // Stack per-expert weights (experts.{i}.{gate,up,down}_proj.weight)
    // into switch_mlp.{gate,up,down}_proj.weight [E, h, d] / [E, d, h]
    final result = <String, MLXArray>{...weights};
    final layerPattern = RegExp(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$');

    // Find unique layer indices
    final layerIndices = <int>{};
    for (final key in weights.keys) {
      final m = layerPattern.firstMatch(key);
      if (m != null) layerIndices.add(int.parse(m.group(1)!));
    }

    for (final layerIdx in layerIndices) {
      for (final proj in ['gate_proj', 'up_proj', 'down_proj']) {
        final expertKeys = <int, MLXArray>{};
        for (final e in weights.entries) {
          final m = RegExp(
                  r'^model\.layers\.' +
                  layerIdx.toString() +
                  r'\.mlp\.experts\.(\d+)\.' +
                  proj +
                  r'\.weight$')
              .firstMatch(e.key);
          if (m != null) {
            expertKeys[int.parse(m.group(1)!)] = e.value;
            result.remove(e.key);
          }
        }
        if (expertKeys.isNotEmpty) {
          final sorted = expertKeys.entries.toList()
            ..sort((a, b) => a.key.compareTo(b.key));
          final stacked =
              stack(_ctx, sorted.map((e) => e.value).toList(), axis: 0);
          result['model.layers.$layerIdx.mlp.switch_mlp.$proj.weight'] = stacked;
        }
      }
    }

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
