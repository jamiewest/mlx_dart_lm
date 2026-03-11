import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class DeepSeekV3Config {
  const DeepSeekV3Config({
    required this.vocabSize,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.moeIntermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    this.nSharedExperts,
    this.nRoutedExperts,
    this.routedScalingFactor = 1.0,
    required this.kvLoraRank,
    required this.qLoraRank,
    required this.qkRopeHeadDim,
    required this.vHeadDim,
    required this.qkNopeHeadDim,
    this.normTopkProb = true,
    this.nGroup = 1,
    this.topkGroup = 1,
    this.numExpertsPerTok = 1,
    this.moeLayerFreq = 1,
    this.firstKDenseReplace = 0,
    this.maxPositionEmbeddings = 2048,
    this.rmsNormEps = 1e-6,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.attentionBias = false,
  });

  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int moeIntermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int? nSharedExperts;
  final int? nRoutedExperts;
  final double routedScalingFactor;
  final int kvLoraRank;
  final int qLoraRank;
  final int qkRopeHeadDim;
  final int vHeadDim;
  final int qkNopeHeadDim;
  final bool normTopkProb;
  final int nGroup;
  final int topkGroup;
  final int numExpertsPerTok;
  final int moeLayerFreq;
  final int firstKDenseReplace;
  final int maxPositionEmbeddings;
  final double rmsNormEps;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final bool attentionBias;

  int get qHeadDim => qkNopeHeadDim + qkRopeHeadDim;

  factory DeepSeekV3Config.fromJson(Map<String, dynamic> j) =>
      DeepSeekV3Config(
        vocabSize: j['vocab_size'] as int,
        hiddenSize: j['hidden_size'] as int,
        intermediateSize: j['intermediate_size'] as int,
        moeIntermediateSize: j['moe_intermediate_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        nSharedExperts: j['n_shared_experts'] as int?,
        nRoutedExperts: j['n_routed_experts'] as int?,
        routedScalingFactor:
            (j['routed_scaling_factor'] as num?)?.toDouble() ?? 1.0,
        kvLoraRank: j['kv_lora_rank'] as int,
        qLoraRank: j['q_lora_rank'] as int,
        qkRopeHeadDim: j['qk_rope_head_dim'] as int,
        vHeadDim: j['v_head_dim'] as int,
        qkNopeHeadDim: j['qk_nope_head_dim'] as int,
        normTopkProb: j['norm_topk_prob'] as bool? ?? true,
        nGroup: j['n_group'] as int? ?? 1,
        topkGroup: j['topk_group'] as int? ?? 1,
        numExpertsPerTok: j['num_experts_per_tok'] as int? ?? 1,
        moeLayerFreq: j['moe_layer_freq'] as int? ?? 1,
        firstKDenseReplace: j['first_k_dense_replace'] as int? ?? 0,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 2048,
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 10000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        attentionBias: j['attention_bias'] as bool? ?? false,
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
// Attention — MLA simplified (kv_b_proj absorbed, standard cache)
// ---------------------------------------------------------------------------

final class _DeepSeekV3Attention extends Module {
  _DeepSeekV3Attention(MLXContext ctx, DeepSeekV3Config cfg)
      : _numHeads = cfg.numAttentionHeads,
        _qkRopeHeadDim = cfg.qkRopeHeadDim,
        _qkNopeHeadDim = cfg.qkNopeHeadDim,
        _vHeadDim = cfg.vHeadDim,
        _kvLoraRank = cfg.kvLoraRank,
        _hasQLoRA = cfg.qLoraRank > 0,
        _scale = _computeScale(cfg),
        qAProj = cfg.qLoraRank > 0
            ? Linear(ctx,
                inFeatures: cfg.hiddenSize,
                outFeatures: cfg.qLoraRank,
                bias: cfg.attentionBias)
            : null,
        qALayernorm = cfg.qLoraRank > 0
            ? RMSNorm(ctx, dims: cfg.qLoraRank, eps: 1e-6)
            : null,
        qBProj = cfg.qLoraRank > 0
            ? Linear(ctx,
                inFeatures: cfg.qLoraRank,
                outFeatures:
                    cfg.numAttentionHeads * (cfg.qkNopeHeadDim + cfg.qkRopeHeadDim),
                bias: false)
            : null,
        qProj = cfg.qLoraRank == 0
            ? Linear(ctx,
                inFeatures: cfg.hiddenSize,
                outFeatures:
                    cfg.numAttentionHeads * (cfg.qkNopeHeadDim + cfg.qkRopeHeadDim),
                bias: false)
            : null,
        kvAProjWithMqa = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.kvLoraRank + cfg.qkRopeHeadDim,
            bias: cfg.attentionBias),
        kvALayernorm = RMSNorm(ctx, dims: cfg.kvLoraRank, eps: 1e-6),
        kvBProj = Linear(ctx,
            inFeatures: cfg.kvLoraRank,
            outFeatures:
                cfg.numAttentionHeads * (cfg.qkNopeHeadDim + cfg.vHeadDim),
            bias: false),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.vHeadDim,
            outFeatures: cfg.hiddenSize,
            bias: cfg.attentionBias),
        rope = initializeRope(
          ctx,
          dims: cfg.qkRopeHeadDim,
          base: cfg.ropeTheta,
          traditional: true,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  static double _computeScale(DeepSeekV3Config cfg) {
    var s = math.pow(cfg.qHeadDim, -0.5).toDouble();
    final rs = cfg.ropeScaling;
    if (rs != null) {
      final mscale = (rs['mscale_all_dim'] as num?)?.toDouble() ?? 0.0;
      if (mscale != 0.0) {
        final factor = (rs['factor'] as num?)?.toDouble() ?? 1.0;
        if (factor > 1.0) {
          final adj = 0.1 * mscale * math.log(factor) + 1.0;
          s = s * adj * adj;
        }
      }
    }
    return s;
  }

  final int _numHeads;
  final int _qkRopeHeadDim;
  final int _qkNopeHeadDim;
  final int _vHeadDim;
  final int _kvLoraRank;
  final bool _hasQLoRA;
  final double _scale;

  Linear? qAProj;
  RMSNorm? qALayernorm;
  Linear? qBProj;
  Linear? qProj;
  Linear kvAProjWithMqa;
  RMSNorm kvALayernorm;
  Linear kvBProj;
  Linear oProj;
  final RopeLayer rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);
    final qHeadDim = _qkNopeHeadDim + _qkRopeHeadDim;

    // --- Query ---
    final MLXArray qRaw;
    if (_hasQLoRA) {
      final qa = qAProj!.call(x);
      final qaNormed = qALayernorm!.call(qa);
      qa.dispose();
      qRaw = qBProj!.call(qaNormed);
      qaNormed.dispose();
    } else {
      qRaw = qProj!.call(x);
    }
    // [B, numHeads, L, qHeadDim]
    final qAll = qRaw
        .reshape([b, s, _numHeads, qHeadDim])
        .transpose([0, 2, 1, 3]);
    qRaw.dispose();

    // Split q into nope and pe parts
    final qNope = qAll.slice(
        start: [0, 0, 0, 0],
        stop: [b, _numHeads, s, _qkNopeHeadDim]); // [B, H, L, nope]
    var qPe = qAll.slice(
        start: [0, 0, 0, _qkNopeHeadDim],
        stop: [b, _numHeads, s, qHeadDim]); // [B, H, L, rope]
    qAll.dispose();

    // --- KV via kv_a_proj + kv_b_proj ---
    final kvAOut = kvAProjWithMqa.call(x); // [B, L, kvLoraRank + qkRopeHeadDim]
    final compressedKv = kvAOut.slice(
        start: [0, 0, 0],
        stop: [b, s, _kvLoraRank]); // [B, L, kvLoraRank]
    final kPeRaw = kvAOut.slice(
        start: [0, 0, _kvLoraRank],
        stop: [b, s, _kvLoraRank + _qkRopeHeadDim]); // [B, L, qkRopeHeadDim]
    kvAOut.dispose();

    final kvLatent = kvALayernorm.call(compressedKv);
    compressedKv.dispose();

    final kvRaw = kvBProj.call(kvLatent); // [B, L, H*(nope+vHead)]
    kvLatent.dispose();
    final kvAll = kvRaw
        .reshape([b, s, _numHeads, _qkNopeHeadDim + _vHeadDim])
        .transpose([0, 2, 1, 3]); // [B, H, L, nope+vHead]
    kvRaw.dispose();

    final kNope = kvAll.slice(
        start: [0, 0, 0, 0],
        stop: [b, _numHeads, s, _qkNopeHeadDim]); // [B, H, L, nope]
    final values = kvAll.slice(
        start: [0, 0, 0, _qkNopeHeadDim],
        stop: [b, _numHeads, s, _qkNopeHeadDim + _vHeadDim]); // [B, H, L, vHead]
    kvAll.dispose();

    // kPe: [B, 1, L, qkRopeHeadDim]
    var kPe = kPeRaw
        .reshape([b, s, 1, _qkRopeHeadDim])
        .transpose([0, 2, 1, 3]);
    kPeRaw.dispose();

    // Apply RoPE
    final offset = cache?.offset ?? 0;
    qPe = rope.call(qPe, offset);
    kPe = rope.call(kPe, offset);

    // Expand kPe from 1 head to numHeads
    final kPeExpanded = kPe.repeat(_numHeads, axis: 1); // [B, H, L, rope]
    kPe.dispose();

    // Concatenate: keys = [kNope, kPeExpanded], queries = [qNope, qPe]
    final keys = concatenate(ctx, [kNope, kPeExpanded], axis: -1);
    kNope.dispose();
    kPeExpanded.dispose();
    final queries = concatenate(ctx, [qNope, qPe], axis: -1);
    qNope.dispose();
    qPe.dispose();

    // Build mask
    MLXArray? mask;
    if (s > 1) {
      mask =
          _additiveCausalMask(ctx, n: s, offset: offset, dtype: queries.dtype);
    }

    // Update KV cache
    MLXArray fullK, fullV;
    if (cache != null) {
      (fullK, fullV) = cache.update(keys, values);
    } else {
      fullK = keys;
      fullV = values;
    }

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: queries,
      keys: fullK,
      values: fullV,
      scale: _scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    mask?.dispose();
    queries.dispose();
    if (cache == null) {
      keys.dispose();
      values.dispose();
    }

    // [B, H, L, vHead] → [B, L, H*vHead]
    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, _numHeads * _vHeadDim]);
    attnOut.dispose();
    final out = oProj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{};
    if (_hasQLoRA) {
      result.addAll(
          qAProj!.parameters().map((k, v) => MapEntry('q_a_proj.$k', v)));
      result.addAll(qALayernorm!
          .parameters()
          .map((k, v) => MapEntry('q_a_layernorm.$k', v)));
      result.addAll(
          qBProj!.parameters().map((k, v) => MapEntry('q_b_proj.$k', v)));
    } else {
      result.addAll(
          qProj!.parameters().map((k, v) => MapEntry('q_proj.$k', v)));
    }
    result.addAll(kvAProjWithMqa.parameters()
        .map((k, v) => MapEntry('kv_a_proj_with_mqa.$k', v)));
    result.addAll(kvALayernorm.parameters()
        .map((k, v) => MapEntry('kv_a_layernorm.$k', v)));
    result.addAll(
        kvBProj.parameters().map((k, v) => MapEntry('kv_b_proj.$k', v)));
    result.addAll(
        oProj.parameters().map((k, v) => MapEntry('o_proj.$k', v)));
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (_hasQLoRA) {
      qAProj!.loadWeights(_scoped(weights, 'q_a_proj'));
      qALayernorm!.loadWeights(_scoped(weights, 'q_a_layernorm'));
      qBProj!.loadWeights(_scoped(weights, 'q_b_proj'));
    } else {
      qProj!.loadWeights(_scoped(weights, 'q_proj'));
    }
    kvAProjWithMqa.loadWeights(_scoped(weights, 'kv_a_proj_with_mqa'));
    kvALayernorm.loadWeights(_scoped(weights, 'kv_a_layernorm'));
    kvBProj.loadWeights(_scoped(weights, 'kv_b_proj'));
    oProj.loadWeights(_scoped(weights, 'o_proj'));
  }

  @override
  void dispose() {
    qAProj?.dispose();
    qALayernorm?.dispose();
    qBProj?.dispose();
    qProj?.dispose();
    kvAProjWithMqa.dispose();
    kvALayernorm.dispose();
    kvBProj.dispose();
    oProj.dispose();
    rope.dispose();
  }
}

// ---------------------------------------------------------------------------
// Dense MLP
// ---------------------------------------------------------------------------

final class _DeepSeekV3DenseMLP extends Module {
  _DeepSeekV3DenseMLP(MLXContext ctx,
      {required int hiddenSize, required int intermediateSize})
      : gateProj = Linear(ctx,
            inFeatures: hiddenSize,
            outFeatures: intermediateSize,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: hiddenSize,
            outFeatures: intermediateSize,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: intermediateSize,
            outFeatures: hiddenSize,
            bias: false);

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

  MLXArray gateProj;
  MLXArray upProj;
  MLXArray downProj;

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
// MoE Gate — group-based top-k with sigmoid scoring
// ---------------------------------------------------------------------------

final class _DeepSeekMoEGate extends Module {
  _DeepSeekMoEGate(MLXContext ctx, DeepSeekV3Config cfg)
      : _hiddenSize = cfg.hiddenSize,
        _nRoutedExperts = cfg.nRoutedExperts!,
        _topK = cfg.numExpertsPerTok,
        _normTopkProb = cfg.normTopkProb,
        _nGroup = cfg.nGroup,
        _topkGroup = cfg.topkGroup,
        _routedScalingFactor = cfg.routedScalingFactor,
        weight = MLXArray.zeros(ctx, [cfg.nRoutedExperts!, cfg.hiddenSize]),
        eScoreCorrectionBias =
            MLXArray.zeros(ctx, [cfg.nRoutedExperts!]);

  final int _hiddenSize;
  final int _nRoutedExperts;
  final int _topK;
  final bool _normTopkProb;
  final int _nGroup;
  final int _topkGroup;
  final double _routedScalingFactor;

  MLXArray weight;             // [E, d]
  MLXArray eScoreCorrectionBias; // [E]

  // Flat-index gather: result[i,j] = src[i, idx[i,j]]
  // src: [rows, cols], idx: [rows, k] → result: [rows, k]
  MLXArray _gatherRows(
      MLXContext ctx, MLXArray src, MLXArray idx) {
    final rows = src.dim(0);
    final cols = src.dim(1);
    final k = idx.dim(1);

    final posRange = MLXArray.arange(ctx, 0.0, rows.toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([rows, 1])
        .repeat(k, axis: 1); // [rows, k]
    final colsArr = MLXArray.int_(ctx, cols);
    final offsets = posRange * colsArr;
    posRange.dispose();
    colsArr.dispose();

    final flatInds = (offsets + idx).reshape([rows * k]);
    offsets.dispose();

    final src1d = src.reshape([rows * cols]);
    final gathered = src1d.take(flatInds, axis: 0).reshape([rows, k]);
    src1d.dispose();
    flatInds.dispose();
    return gathered;
  }

  (MLXArray inds, MLXArray scores) call(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final bl = b * s;
    final e = _nRoutedExperts;
    final k = _topK;

    final xFlat = x.reshape([bl, _hiddenSize]);
    final logits = xFlat.matmul(weight.T); // [bl, E]
    xFlat.dispose();

    final sigScores = logits.sigmoid(); // [bl, E] — orig_scores for final
    logits.dispose();

    final biasExpanded = eScoreCorrectionBias.reshape([1, e]);
    final scoresForChoice = sigScores + biasExpanded; // [bl, E]
    biasExpanded.dispose();

    final MLXArray topKInds; // [bl, k]

    if (_nGroup > 1 && _topkGroup < _nGroup) {
      final g = e ~/ _nGroup; // experts per group

      // Compute per-group top-2 score sum
      // groupedScores: [bl, nGroup, G]
      final groupedScores =
          scoresForChoice.reshape([bl, _nGroup, g]);

      // For each group, sum the top-2 scores via argsort
      // Flatten to [bl*nGroup, G], argsort desc, take top-2, gather, sum
      final groupedFlat = groupedScores.reshape([bl * _nGroup, g]);
      final negGrouped = groupedFlat * MLXArray.float_(ctx, -1.0);
      final sortedG = negGrouped.argsort(axis: -1); // [bl*nGroup, G]
      negGrouped.dispose();
      final top2Inds = sortedG.slice(
          start: [0, 0], stop: [bl * _nGroup, 2]); // [bl*nGroup, 2]
      sortedG.dispose();

      // Gather top-2 values
      final top2Vals = _gatherRows(ctx, groupedFlat, top2Inds); // [bl*nGroup, 2]
      groupedFlat.dispose();
      top2Inds.dispose();
      groupedScores.dispose();

      // Sum top-2 per group → group score
      final groupScores =
          top2Vals.sum(axis: -1).reshape([bl, _nGroup]); // [bl, nGroup]
      top2Vals.dispose();

      // Select top topkGroup groups
      final negGroupScores = groupScores * MLXArray.float_(ctx, -1.0);
      groupScores.dispose();
      final sortedGroupInds = negGroupScores.argsort(axis: -1); // [bl, nGroup]
      negGroupScores.dispose();
      final topGroupInds = sortedGroupInds.slice(
          start: [0, 0], stop: [bl, _topkGroup]); // [bl, topkGroup]
      sortedGroupInds.dispose();

      // Compute eligible expert indices: topGroupInds * G + arange(G)
      // topGroupBase: [bl, topkGroup] * G → expand to [bl, topkGroup, G]
      final gArr = MLXArray.int_(ctx, g);
      final topGroupBase = topGroupInds * gArr; // [bl, topkGroup]
      gArr.dispose();
      final gRange = MLXArray.arange(ctx, 0.0, g.toDouble(), 1.0,
              dtype: MLXDtype.int32)
          .reshape([1, 1, g]); // [1, 1, G]
      final eligibleExperts =
          (topGroupBase.expandDims(2) + gRange).reshape([bl, _topkGroup * g]);
      topGroupInds.dispose();
      topGroupBase.dispose();
      gRange.dispose();

      // Gather scoresForChoice at eligibleExperts → [bl, topkGroup*G]
      final eligibleScores =
          _gatherRows(ctx, scoresForChoice, eligibleExperts); // [bl, topkGroup*G]

      // Select top-k from eligible (using scores with bias for ranking)
      final negEligible = eligibleScores * MLXArray.float_(ctx, -1.0);
      eligibleScores.dispose();
      final sortedLocalInds = negEligible.argsort(axis: -1);
      negEligible.dispose();
      final localTopK = sortedLocalInds.slice(
          start: [0, 0], stop: [bl, k]); // [bl, k] — local indices into eligibleExperts
      sortedLocalInds.dispose();

      // Map local indices → global expert indices
      topKInds =
          _gatherRows(ctx, eligibleExperts, localTopK); // [bl, k] global expert inds
      eligibleExperts.dispose();
      localTopK.dispose();
    } else {
      // Simple top-k without group filtering
      final negScores = scoresForChoice * MLXArray.float_(ctx, -1.0);
      final sortedInds = negScores.argsort(axis: -1);
      negScores.dispose();
      topKInds = sortedInds.slice(start: [0, 0], stop: [bl, k]);
      sortedInds.dispose();
    }
    scoresForChoice.dispose();

    // Gather orig_scores (sigmoid without bias) at topKInds
    var topKScores = _gatherRows(ctx, sigScores, topKInds); // [bl, k]
    sigScores.dispose();

    // Normalize
    if (k > 1 && _normTopkProb) {
      final sumScores = topKScores.sum(axis: -1, keepdims: true) +
          MLXArray.float_(ctx, 1e-20);
      final normalized = topKScores / sumScores;
      sumScores.dispose();
      topKScores.dispose();
      topKScores = normalized;
    }

    // Scale
    final scaled = topKScores * MLXArray.float_(ctx, _routedScalingFactor);
    topKScores.dispose();

    return (
      topKInds.reshape([b, s, k]),
      scaled.reshape([b, s, k]),
    );
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

final class _DeepSeekV3MoE extends Module {
  _DeepSeekV3MoE(MLXContext ctx, DeepSeekV3Config cfg)
      : gate = _DeepSeekMoEGate(ctx, cfg),
        switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.nRoutedExperts!),
        sharedExperts = cfg.nSharedExperts != null
            ? _DeepSeekV3DenseMLP(ctx,
                hiddenSize: cfg.hiddenSize,
                intermediateSize:
                    cfg.moeIntermediateSize * cfg.nSharedExperts!)
            : null;

  _DeepSeekMoEGate gate;
  _SwitchGLU switchMlp;
  _DeepSeekV3DenseMLP? sharedExperts;

  MLXArray call(MLXContext ctx, MLXArray x) {
    final (inds, scores) = gate.call(ctx, x);
    final routedOut = switchMlp.call(ctx, x, inds, scores);
    inds.dispose();
    scores.dispose();

    final MLXArray out;
    if (sharedExperts != null) {
      final sharedOut = sharedExperts!.call(x);
      out = routedOut + sharedOut;
      routedOut.dispose();
      sharedOut.dispose();
    } else {
      out = routedOut;
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
      result.addAll(sharedExperts!
          .parameters()
          .map((k, v) => MapEntry('shared_experts.$k', v)));
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
// Decoder layer
// ---------------------------------------------------------------------------

sealed class _DSV3MLP extends Module {
  MLXArray call(MLXContext ctx, MLXArray x);
}

final class _DSV3DenseLayer extends _DSV3MLP {
  _DSV3DenseLayer(_DeepSeekV3DenseMLP mlp) : _mlp = mlp;
  final _DeepSeekV3DenseMLP _mlp;

  @override
  MLXArray call(MLXContext ctx, MLXArray x) => _mlp.call(x);

  @override
  Map<String, MLXArray> parameters() => _mlp.parameters();

  @override
  void loadWeights(Map<String, MLXArray> weights) =>
      _mlp.loadWeights(weights);

  @override
  void dispose() => _mlp.dispose();
}

final class _DSV3MoELayer extends _DSV3MLP {
  _DSV3MoELayer(_DeepSeekV3MoE moe) : _moe = moe;
  final _DeepSeekV3MoE _moe;

  @override
  MLXArray call(MLXContext ctx, MLXArray x) => _moe.call(ctx, x);

  @override
  Map<String, MLXArray> parameters() => _moe.parameters();

  @override
  void loadWeights(Map<String, MLXArray> weights) =>
      _moe.loadWeights(weights);

  @override
  void dispose() => _moe.dispose();
}

final class _DeepSeekV3DecoderLayer extends Module {
  _DeepSeekV3DecoderLayer(MLXContext ctx, DeepSeekV3Config cfg, int layerIdx)
      : selfAttn = _DeepSeekV3Attention(ctx, cfg),
        mlp = _isMoELayer(cfg, layerIdx)
            ? _DSV3MoELayer(_DeepSeekV3MoE(ctx, cfg))
            : _DSV3DenseLayer(_DeepSeekV3DenseMLP(ctx,
                hiddenSize: cfg.hiddenSize,
                intermediateSize: cfg.intermediateSize)),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  static bool _isMoELayer(DeepSeekV3Config cfg, int idx) =>
      cfg.nRoutedExperts != null &&
      idx >= cfg.firstKDenseReplace &&
      idx % cfg.moeLayerFreq == 0;

  _DeepSeekV3Attention selfAttn;
  _DSV3MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final h = x + attnOut;
    attnOut.dispose();

    final normed2 = postAttentionLayernorm.call(h);
    final mlpOut = mlp.call(ctx, normed2);
    normed2.dispose();
    final out = h + mlpOut;
    mlpOut.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in selfAttn.parameters().entries)
          'self_attn.${e.key}': e.value,
        for (final e in mlp.parameters().entries) 'mlp.${e.key}': e.value,
        for (final e in inputLayernorm.parameters().entries)
          'input_layernorm.${e.key}': e.value,
        for (final e in postAttentionLayernorm.parameters().entries)
          'post_attention_layernorm.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
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

final class _DeepSeekV3InnerModel extends Module {
  _DeepSeekV3InnerModel(MLXContext ctx, DeepSeekV3Config cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers,
            (i) => _DeepSeekV3DecoderLayer(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_DeepSeekV3DecoderLayer> layers;
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
// Public DeepSeekV3Model
// ---------------------------------------------------------------------------

final class DeepSeekV3Model extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  DeepSeekV3Model(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _DeepSeekV3InnerModel(ctx, config),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final DeepSeekV3Config config;
  final _DeepSeekV3InnerModel _model;
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
  List<int> get kvHeads {
    // KV cache stores [kNope+kPe] per head, so numHeads heads per layer
    return List.filled(config.numHiddenLayers, config.numAttentionHeads);
  }

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = Map<String, MLXArray>.from(weights);

    // FP8 dequantization: weight_scale_inv keys indicate FP8 weights
    // Each weight has a corresponding {key}_scale_inv with block scaling factors
    final fp8Keys = result.keys
        .where((k) => k.endsWith('weight_scale_inv'))
        .toList();
    for (final scaleKey in fp8Keys) {
      final weightKey = scaleKey.replaceAll('_scale_inv', '');
      if (result.containsKey(weightKey)) {
        final weight = result[weightKey]!;
        final scaleInv = result.remove(scaleKey)!;
        // Block-wise FP8 dequant: scale blocks of 128
        const bs = 128;
        final m = weight.dim(0);
        final n = weight.dim(1);
        final padBottom = (bs - m % bs) % bs;
        final padSide = (bs - n % bs) % bs;
        final padded = weight.pad(
          axes: [0, 1],
          lowPad: [0, 0],
          highPad: [padBottom, padSide],
        );
        final reshaped = padded.reshape([
          (m + padBottom) ~/ bs,
          bs,
          (n + padSide) ~/ bs,
          bs,
        ]);
        padded.dispose();
        // scaleInv shape: [(m+padBottom)//bs, (n+padSide)//bs]
        final scaleExpanded = scaleInv
            .expandDims(1)
            .expandDims(3); // [..., 1, ..., 1]
        final scaled = reshaped * scaleExpanded;
        reshaped.dispose();
        scaleExpanded.dispose();
        scaleInv.dispose();
        final full = scaled.reshape([m + padBottom, n + padSide]);
        scaled.dispose();
        final dequantized =
            full.slice(start: [0, 0], stop: [m, n]).astype(MLXDtype.bfloat16);
        full.dispose();
        result[weightKey] = dequantized;
      }
    }

    // Stack per-expert weights: {gate_proj,down_proj,up_proj}
    final nExperts = config.nRoutedExperts;
    if (nExperts != null) {
      const probe = 'model.layers.0.mlp.experts.0.gate_proj.weight';
      if (result.containsKey(probe)) {
        for (var l = 0; l < config.numHiddenLayers; l++) {
          final prefix = 'model.layers.$l.mlp';
          for (final proj in ['gate_proj', 'down_proj', 'up_proj']) {
            for (final k in ['weight', 'scales', 'biases']) {
              if (!result.containsKey('$prefix.experts.0.$proj.$k')) continue;
              final tensors = [
                for (var e = 0; e < nExperts; e++)
                  result.remove('$prefix.experts.$e.$proj.$k')!,
              ];
              result['$prefix.switch_mlp.$proj.$k'] =
                  stack(_ctx, tensors, axis: 0);
            }
          }
        }
      }
    }

    // Filter out MTP (multi-token prediction) layers and rotary inv_freq
    result.removeWhere((k, _) =>
        k.startsWith('model.layers.${config.numHiddenLayers}.') ||
        k.contains('rotary_emb.inv_freq'));

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
