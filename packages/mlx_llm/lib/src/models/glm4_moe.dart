import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class GLM4MoEConfig {
  const GLM4MoEConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.moeIntermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    required this.maxPositionEmbeddings,
    this.rmsNormEps = 1e-5,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.partialRotaryFactor = 1.0,
    this.useQkNorm = false,
    this.attentionBias = false,
    this.tieWordEmbeddings = false,
    this.nRoutedExperts,
    this.nSharedExperts,
    this.numExpertsPerTok = 1,
    this.firstKDenseReplace = 0,
    this.nGroup = 1,
    this.topkGroup = 1,
    this.normTopkProb = false,
    this.routedScalingFactor = 1.0,
    this.scoringFunc = 'sigmoid',
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int moeIntermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final int maxPositionEmbeddings;
  final double rmsNormEps;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final double partialRotaryFactor;
  final bool useQkNorm;
  final bool attentionBias;
  final bool tieWordEmbeddings;
  final int? nRoutedExperts;
  final int? nSharedExperts;
  final int numExpertsPerTok;
  final int firstKDenseReplace;
  final int nGroup;
  final int topkGroup;
  final bool normTopkProb;
  final double routedScalingFactor;
  final String scoringFunc;

  int get effectiveHeadDim =>
      headDim > 0 ? headDim : hiddenSize ~/ numAttentionHeads;
  int get ropeDims =>
      (partialRotaryFactor * effectiveHeadDim).round().clamp(1, effectiveHeadDim);

  factory GLM4MoEConfig.fromJson(Map<String, dynamic> j) => GLM4MoEConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        moeIntermediateSize: j['moe_intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        headDim: j['head_dim'] as int? ?? 0,
        vocabSize: j['vocab_size'] as int,
        maxPositionEmbeddings: j['max_position_embeddings'] as int,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 1.0,
        useQkNorm: j['use_qk_norm'] as bool? ?? false,
        attentionBias: j['attention_bias'] as bool? ?? false,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        nRoutedExperts: j['n_routed_experts'] as int?,
        nSharedExperts: j['n_shared_experts'] as int?,
        numExpertsPerTok: j['num_experts_per_tok'] as int? ?? 1,
        firstKDenseReplace: j['first_k_dense_replace'] as int? ?? 0,
        nGroup: j['n_group'] as int? ?? 1,
        topkGroup: j['topk_group'] as int? ?? 1,
        normTopkProb: j['norm_topk_prob'] as bool? ?? false,
        routedScalingFactor:
            (j['routed_scaling_factor'] as num?)?.toDouble() ?? 1.0,
        scoringFunc: j['scoring_func'] as String? ?? 'sigmoid',
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
  _SwitchGLU(MLXContext ctx,
      {required int inputDims,
      required int hiddenDims,
      required int numExperts})
      : _inputDims = inputDims,
        _numExperts = numExperts,
        gateProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        upProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        downProj = MLXArray.zeros(ctx, [numExperts, inputDims, hiddenDims]);

  final int _inputDims;
  final int _numExperts;

  MLXArray gateProj;
  MLXArray upProj;
  MLXArray downProj;

  MLXArray _computeAll(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final d = _inputDims;
    final e = _numExperts;
    final bl = b * s;
    final xFlat = x.reshape([bl, d]);

    final gw = gateProj.reshape([e * gateProj.dim(1), d]);
    final gateAll = xFlat.matmul(gw.T).reshape([bl, e, gateProj.dim(1)]);
    gw.dispose();

    final uw = upProj.reshape([e * upProj.dim(1), d]);
    final upAll = xFlat.matmul(uw.T).reshape([bl, e, upProj.dim(1)]);
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
    return outT.transpose([1, 0, 2]).reshape([b, s, e, d]);
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
    final offsets = posRange * MLXArray.int_(ctx, e);
    posRange.dispose();

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
    final weighted = selectedBSKD * scores.expandDims(3);
    selectedBSKD.dispose();

    final out = weighted.sum(axis: 2);
    weighted.dispose();
    return out;
  }

  // Returns [b, s, d] weighted sum.
  MLXArray call(
      MLXContext ctx, MLXArray x, MLXArray inds, MLXArray scores) {
    final allOuts = _computeAll(ctx, x);
    final out = _gatherAndSum(ctx, allOuts, inds, scores);
    allOuts.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        'gate_proj': gateProj,
        'up_proj': upProj,
        'down_proj': downProj,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights.containsKey('gate_proj')) {
      gateProj.dispose();
      gateProj = weights['gate_proj']!;
    }
    if (weights.containsKey('up_proj')) {
      upProj.dispose();
      upProj = weights['up_proj']!;
    }
    if (weights.containsKey('down_proj')) {
      downProj.dispose();
      downProj = weights['down_proj']!;
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
// Dense MLP
// ---------------------------------------------------------------------------

final class _GLM4MoEMLP extends Module {
  _GLM4MoEMLP(MLXContext ctx, {required int inDim, required int hiddenDim})
      : gateProj =
            Linear(ctx, inFeatures: inDim, outFeatures: hiddenDim, bias: false),
        upProj =
            Linear(ctx, inFeatures: inDim, outFeatures: hiddenDim, bias: false),
        downProj =
            Linear(ctx, inFeatures: hiddenDim, outFeatures: inDim, bias: false);

  Linear gateProj;
  Linear upProj;
  Linear downProj;

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
        for (final e in upProj.parameters().entries)
          'up_proj.${e.key}': e.value,
        for (final e in downProj.parameters().entries)
          'down_proj.${e.key}': e.value,
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
// MoE Gate — group-based top-k with sigmoid/softmax scoring + correction bias
// ---------------------------------------------------------------------------

final class _GLM4MoEGate extends Module {
  _GLM4MoEGate(MLXContext ctx, GLM4MoEConfig cfg)
      : _nExperts = cfg.nRoutedExperts!,
        _topK = cfg.numExpertsPerTok,
        _nGroup = cfg.nGroup,
        _topkGroup = cfg.topkGroup,
        _normTopkProb = cfg.normTopkProb,
        _routedScalingFactor = cfg.routedScalingFactor,
        _useSigmoid = cfg.scoringFunc == 'sigmoid',
        weight = MLXArray.zeros(ctx, [cfg.nRoutedExperts!, cfg.hiddenSize]),
        eScoreCorrectionBias =
            MLXArray.zeros(ctx, [cfg.nRoutedExperts!]);

  final int _nExperts;
  final int _topK;
  final int _nGroup;
  final int _topkGroup;
  final bool _normTopkProb;
  final double _routedScalingFactor;
  final bool _useSigmoid;

  MLXArray weight;
  MLXArray eScoreCorrectionBias;

  /// Flat-index gather: result[i,j] = src[i, idx[i,j]]
  MLXArray _gatherRows(MLXContext ctx, MLXArray src, MLXArray idx) {
    final rows = src.dim(0);
    final cols = src.dim(1);
    final k = idx.dim(1);
    final posRange = MLXArray.arange(ctx, 0.0, rows.toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([rows, 1])
        .repeat(k, axis: 1);
    final offsets = posRange * MLXArray.int_(ctx, cols);
    posRange.dispose();
    final flatInds = (offsets + idx).reshape([rows * k]);
    offsets.dispose();
    final gathered =
        src.reshape([rows * cols]).take(flatInds, axis: 0).reshape([rows, k]);
    flatInds.dispose();
    return gathered;
  }

  /// Returns `(inds, scores)` both shaped [bl, k].
  (MLXArray inds, MLXArray scores) call(MLXContext ctx, MLXArray x) {
    final bl = x.dim(0);
    final e = _nExperts;
    final k = _topK;

    final logits = x.matmul(weight.T); // [bl, E]
    final scores =
        _useSigmoid ? logits.sigmoid() : logits.softmax(axis: -1);
    logits.dispose();

    final bias = eScoreCorrectionBias.reshape([1, e]);
    final scoresForChoice = scores + bias;
    bias.dispose();

    final MLXArray topKInds;

    if (_nGroup > 1 && _topkGroup < _nGroup) {
      final g = e ~/ _nGroup;

      // Per-group top-2 sum
      final groupedFlat = scoresForChoice.reshape([bl * _nGroup, g]);
      final negGrouped = groupedFlat * MLXArray.float_(ctx, -1.0);
      final sortedG = negGrouped.argsort(axis: -1);
      negGrouped.dispose();
      final top2Inds = sortedG.slice(start: [0, 0], stop: [bl * _nGroup, 2]);
      sortedG.dispose();
      final top2Vals = _gatherRows(ctx, groupedFlat, top2Inds);
      groupedFlat.dispose();
      top2Inds.dispose();

      final groupScores = top2Vals.sum(axis: -1).reshape([bl, _nGroup]);
      top2Vals.dispose();

      // Select top topkGroup groups
      final negGroupScores = groupScores * MLXArray.float_(ctx, -1.0);
      groupScores.dispose();
      final sortedGroupInds = negGroupScores.argsort(axis: -1);
      negGroupScores.dispose();
      final topGroupInds =
          sortedGroupInds.slice(start: [0, 0], stop: [bl, _topkGroup]);
      sortedGroupInds.dispose();

      // Eligible experts: topGroupInds * G + arange(G)
      final topGroupBase = topGroupInds * MLXArray.int_(ctx, g);
      final gRange = MLXArray.arange(ctx, 0.0, g.toDouble(), 1.0,
              dtype: MLXDtype.int32)
          .reshape([1, 1, g]);
      final eligibleExperts =
          (topGroupBase.expandDims(2) + gRange).reshape([bl, _topkGroup * g]);
      topGroupInds.dispose();
      topGroupBase.dispose();
      gRange.dispose();

      final eligibleScores = _gatherRows(ctx, scoresForChoice, eligibleExperts);
      final negEligible = eligibleScores * MLXArray.float_(ctx, -1.0);
      eligibleScores.dispose();
      final sortedLocal = negEligible.argsort(axis: -1);
      negEligible.dispose();
      final localTopK = sortedLocal.slice(start: [0, 0], stop: [bl, k]);
      sortedLocal.dispose();
      topKInds = _gatherRows(ctx, eligibleExperts, localTopK);
      eligibleExperts.dispose();
      localTopK.dispose();
    } else {
      final negScores = scoresForChoice * MLXArray.float_(ctx, -1.0);
      final sortedInds = negScores.argsort(axis: -1);
      negScores.dispose();
      topKInds = sortedInds.slice(start: [0, 0], stop: [bl, k]);
      sortedInds.dispose();
    }
    scoresForChoice.dispose();

    // Gather original scores (no bias) at selected indices
    var topKScores = _gatherRows(ctx, scores, topKInds);
    scores.dispose();

    if (k > 1 && _normTopkProb) {
      final sumS = topKScores.sum(axis: -1, keepdims: true) +
          MLXArray.float_(ctx, 1e-20);
      final normed = topKScores / sumS;
      sumS.dispose();
      topKScores.dispose();
      topKScores = normed;
    }
    final scaled = topKScores * MLXArray.float_(ctx, _routedScalingFactor);
    topKScores.dispose();
    return (topKInds, scaled);
  }

  @override
  Map<String, MLXArray> parameters() => {
        'weight': weight,
        'e_score_correction_bias': eScoreCorrectionBias,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights.containsKey('weight')) {
      weight.dispose();
      weight = weights['weight']!;
    }
    if (weights.containsKey('e_score_correction_bias')) {
      eScoreCorrectionBias.dispose();
      eScoreCorrectionBias = weights['e_score_correction_bias']!;
    }
  }

  @override
  void dispose() {
    weight.dispose();
    eScoreCorrectionBias.dispose();
  }
}

// ---------------------------------------------------------------------------
// MoE block — gate + SwitchGLU + optional shared experts
// ---------------------------------------------------------------------------

final class _GLM4MoEBlock extends Module {
  _GLM4MoEBlock(MLXContext ctx, GLM4MoEConfig cfg)
      : gate = _GLM4MoEGate(ctx, cfg),
        switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.nRoutedExperts!),
        sharedExperts = (cfg.nSharedExperts != null && cfg.nSharedExperts! > 0)
            ? _GLM4MoEMLP(ctx,
                inDim: cfg.hiddenSize,
                hiddenDim: cfg.moeIntermediateSize * cfg.nSharedExperts!)
            : null;

  _GLM4MoEGate gate;
  _SwitchGLU switchMlp;
  _GLM4MoEMLP? sharedExperts;

  MLXArray call(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final bl = b * s;
    final xFlat = x.reshape([bl, x.dim(2)]);

    final (indsFlat, scoresFlat) = gate.call(ctx, xFlat); // [bl, k]
    xFlat.dispose();

    // Reshape to [b, s, k] for SwitchGLU
    final inds3d = indsFlat.reshape([b, s, indsFlat.dim(1)]);
    final scores3d = scoresFlat.reshape([b, s, scoresFlat.dim(1)]);
    indsFlat.dispose();
    scoresFlat.dispose();

    // SwitchGLU returns [b, s, d] already weighted
    var out = switchMlp.call(ctx, x, inds3d, scores3d);
    inds3d.dispose();
    scores3d.dispose();

    if (sharedExperts != null) {
      final shared = sharedExperts!.call(x);
      final combined = out + shared;
      shared.dispose();
      out.dispose();
      out = combined;
    }
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in gate.parameters().entries) 'gate.${e.key}': e.value,
      for (final e in switchMlp.parameters().entries)
        'switch_mlp.${e.key}': e.value,
    };
    if (sharedExperts != null) {
      for (final e in sharedExperts!.parameters().entries) {
        result['shared_experts.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gate.loadWeights(_scoped(weights, 'gate'));
    switchMlp.loadWeights(_scoped(weights, 'switch_mlp'));
    sharedExperts?.loadWeights(_scoped(weights, 'shared_experts'));
  }

  @override
  void dispose() {
    gate.dispose();
    switchMlp.dispose();
    sharedExperts?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — optional per-head QK-norm, partial RoPE
// ---------------------------------------------------------------------------

final class _GLM4MoEAttention extends Module {
  _GLM4MoEAttention(MLXContext ctx, GLM4MoEConfig cfg)
      : wq = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        wk = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        wv = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.effectiveHeadDim,
            bias: cfg.attentionBias),
        wo = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.effectiveHeadDim,
            outFeatures: cfg.hiddenSize,
            bias: false),
        qNorm = cfg.useQkNorm
            ? RMSNorm(ctx,
                dims: cfg.effectiveHeadDim, eps: cfg.rmsNormEps)
            : null,
        kNorm = cfg.useQkNorm
            ? RMSNorm(ctx,
                dims: cfg.effectiveHeadDim, eps: cfg.rmsNormEps)
            : null,
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.effectiveHeadDim,
        scale = 1.0 / math.sqrt(cfg.effectiveHeadDim.toDouble()),
        _rope = initializeRope(
          ctx,
          dims: cfg.ropeDims,
          traditional: false,
          base: cfg.ropeTheta,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  Linear wq;
  Linear wk;
  Linear wv;
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

    // Per-head QK-norm applied before transpose
    var q = wq.call(x).reshape([b, s, numHeads, headDim]);
    var k = wk.call(x).reshape([b, s, numKVHeads, headDim]);
    final v = wv
        .call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);

    if (qNorm != null) {
      final qn = qNorm!.call(q);
      q.dispose();
      q = qn;
    }
    if (kNorm != null) {
      final kn = kNorm!.call(k);
      k.dispose();
      k = kn;
    }

    q = q.transpose([0, 2, 1, 3]);
    k = k.transpose([0, 2, 1, 3]);

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
    final out = wo.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in wq.parameters().entries) 'q_proj.${e.key}': e.value,
      for (final e in wk.parameters().entries) 'k_proj.${e.key}': e.value,
      for (final e in wv.parameters().entries) 'v_proj.${e.key}': e.value,
      for (final e in wo.parameters().entries) 'o_proj.${e.key}': e.value,
    };
    if (qNorm != null) {
      for (final e in qNorm!.parameters().entries) {
        result['q_norm.${e.key}'] = e.value;
      }
    }
    if (kNorm != null) {
      for (final e in kNorm!.parameters().entries) {
        result['k_norm.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    wq.loadWeights(_scoped(weights, 'q_proj'));
    wk.loadWeights(_scoped(weights, 'k_proj'));
    wv.loadWeights(_scoped(weights, 'v_proj'));
    wo.loadWeights(_scoped(weights, 'o_proj'));
    qNorm?.loadWeights(_scoped(weights, 'q_norm'));
    kNorm?.loadWeights(_scoped(weights, 'k_norm'));
  }

  @override
  void dispose() {
    wq.dispose();
    wk.dispose();
    wv.dispose();
    wo.dispose();
    qNorm?.dispose();
    kNorm?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder block — standard pre-norm; dense or MoE per layer
// ---------------------------------------------------------------------------

final class _GLM4MoEDecoderLayer extends Module {
  _GLM4MoEDecoderLayer(MLXContext ctx, GLM4MoEConfig cfg, int layerIdx)
      : selfAttn = _GLM4MoEAttention(ctx, cfg),
        isMoE =
            cfg.nRoutedExperts != null && layerIdx >= cfg.firstKDenseReplace,
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps) {
    if (cfg.nRoutedExperts != null && layerIdx >= cfg.firstKDenseReplace) {
      moe = _GLM4MoEBlock(ctx, cfg);
    } else {
      denseMlp = _GLM4MoEMLP(ctx,
          inDim: cfg.hiddenSize, hiddenDim: cfg.intermediateSize);
    }
  }

  _GLM4MoEAttention selfAttn;
  final bool isMoE;
  _GLM4MoEBlock? moe;
  _GLM4MoEMLP? denseMlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final r = selfAttn.call(ctx, inputLayernorm.call(x), cache);
    final h = x + r;
    r.dispose();
    final normedH = postAttentionLayernorm.call(h);
    final mlpOut = isMoE
        ? moe!.call(ctx, normedH)
        : denseMlp!.call(normedH);
    normedH.dispose();
    final out = h + mlpOut;
    mlpOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        if (isMoE)
          for (final e in moe!.parameters().entries) 'mlp.${e.key}': e.value
        else
          for (final e in denseMlp!.parameters().entries)
            'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    if (isMoE) {
      moe!.loadWeights(_scoped(weights, 'mlp'));
    } else {
      denseMlp!.loadWeights(_scoped(weights, 'mlp'));
    }
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    moe?.dispose();
    denseMlp?.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model
// ---------------------------------------------------------------------------

final class _GLM4MoEInnerModel extends Module {
  _GLM4MoEInnerModel(MLXContext ctx, GLM4MoEConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(
            cfg.numHiddenLayers, (i) => _GLM4MoEDecoderLayer(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_GLM4MoEDecoderLayer> layers;
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
// Public GLM4MoEModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// GLM4 Mixture-of-Experts language model.
///
/// GLM4-style attention (optional per-head QK-norm, partial RoPE) with
/// group-based top-k expert routing using sigmoid/softmax + correction bias.
/// Dense MLP for layers < `firstKDenseReplace`, MoE thereafter.
///
/// Mirrors `GLM4MoEModel` from mlx-swift-lm.
final class GLM4MoEModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  GLM4MoEModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _GLM4MoEInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final GLM4MoEConfig config;
  final _GLM4MoEInnerModel _model;
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
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = <String, MLXArray>{...weights};

    if (config.tieWordEmbeddings) {
      result.remove('lm_head.weight');
    }

    // Remove MTP layers (model.layers.<numHiddenLayers>.*)
    final mptPrefix = 'model.layers.${config.numHiddenLayers}.';
    result.removeWhere((k, _) => k.startsWith(mptPrefix));

    final nExperts = config.nRoutedExperts;
    if (nExperts != null) {
      for (var l = 0; l < config.numHiddenLayers; l++) {
        final prefix = 'model.layers.$l.mlp';
        for (final n in ['gate_proj', 'down_proj', 'up_proj']) {
          for (final k in ['weight', 'scales', 'biases']) {
            final key0 = '$prefix.experts.0.$n.$k';
            if (!result.containsKey(key0)) continue;
            final tensors = [
              for (var e = 0; e < nExperts; e++)
                result.remove('$prefix.experts.$e.$n.$k')!
            ];
            result['$prefix.switch_mlp.$n.$k'] =
                stack(_ctx, tensors, axis: 0);
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
