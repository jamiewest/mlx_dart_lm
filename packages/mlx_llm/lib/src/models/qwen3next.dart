import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Qwen3NextConfig {
  const Qwen3NextConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.linearNumValueHeads,
    required this.linearNumKeyHeads,
    required this.linearKeyHeadDim,
    required this.linearValueHeadDim,
    required this.linearConvKernelDim,
    required this.numExperts,
    required this.numExpertsPerTok,
    required this.decoderSparseStep,
    required this.sharedExpertIntermediateSize,
    required this.moeIntermediateSize,
    this.mlpOnlyLayers = const [],
    this.rmsNormEps = 1e-6,
    required this.vocabSize,
    this.ropeTheta = 1000000.0,
    this.partialRotaryFactor = 1.0,
    this.maxPositionEmbeddings = 32768,
    this.normTopkProb = false,
    this.tieWordEmbeddings = false,
    this.attentionBias = false,
    this.headDim,
    this.ropeScaling,
    this.fullAttentionInterval = 4,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int linearNumValueHeads;
  final int linearNumKeyHeads;
  final int linearKeyHeadDim;
  final int linearValueHeadDim;
  final int linearConvKernelDim;
  final int numExperts;
  final int numExpertsPerTok;
  final int decoderSparseStep;
  final int sharedExpertIntermediateSize;
  final int moeIntermediateSize;
  final List<int> mlpOnlyLayers;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final double partialRotaryFactor;
  final int maxPositionEmbeddings;
  final bool normTopkProb;
  final bool tieWordEmbeddings;
  final bool attentionBias;
  final int? headDim;
  final Map<String, dynamic>? ropeScaling;
  final int fullAttentionInterval;

  int get attnHeadDim => headDim ?? (hiddenSize ~/ numAttentionHeads);
  int get keyDim => linearNumKeyHeads * linearKeyHeadDim;
  int get valueDim => linearNumValueHeads * linearValueHeadDim;
  int get convDim => keyDim * 2 + valueDim;

  bool isLinearLayer(int idx) => (idx + 1) % fullAttentionInterval != 0;
  bool isMoELayer(int idx) =>
      !mlpOnlyLayers.contains(idx) &&
      numExperts > 0 &&
      (idx + 1) % decoderSparseStep == 0;

  factory Qwen3NextConfig.fromJson(Map<String, dynamic> j) => Qwen3NextConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        linearNumValueHeads: j['linear_num_value_heads'] as int,
        linearNumKeyHeads: j['linear_num_key_heads'] as int,
        linearKeyHeadDim: j['linear_key_head_dim'] as int,
        linearValueHeadDim: j['linear_value_head_dim'] as int,
        linearConvKernelDim: j['linear_conv_kernel_dim'] as int,
        numExperts: j['num_experts'] as int? ?? 0,
        numExpertsPerTok: j['num_experts_per_tok'] as int? ?? 1,
        decoderSparseStep: j['decoder_sparse_step'] as int? ?? 1,
        sharedExpertIntermediateSize:
            j['shared_expert_intermediate_size'] as int? ?? 0,
        moeIntermediateSize: j['moe_intermediate_size'] as int? ?? 0,
        mlpOnlyLayers:
            (j['mlp_only_layers'] as List?)?.map((v) => v as int).toList() ??
                const [],
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
        partialRotaryFactor:
            (j['partial_rotary_factor'] as num?)?.toDouble() ?? 1.0,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 32768,
        normTopkProb: j['norm_topk_prob'] as bool? ?? false,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        attentionBias: j['attention_bias'] as bool? ?? false,
        headDim: j['head_dim'] as int?,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        fullAttentionInterval:
            j['full_attention_interval'] as int? ?? 4,
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
MLXArray _rmsNormNoWeight(MLXContext ctx, MLXArray x, double eps) {
  final sq = x * x;
  final variance = sq.mean(axis: -1, keepdims: true);
  sq.dispose();
  final norm = x / (variance + MLXArray.float_(ctx, eps)).sqrt();
  variance.dispose();
  return norm;
}

/// Numerically stable softplus: log(1 + exp(x)).
MLXArray _softplus(MLXContext ctx, MLXArray x) =>
    (MLXArray.float_(ctx, 1.0) + x.exp()).log();

// ---------------------------------------------------------------------------
// Depthwise Conv1d module — groups == convDim
//
// Weight stored as [convDim, kernelSize, 1] in MLX format.
// Checkpoint may ship as [convDim, 1, kernelSize] (PyTorch) — sanitize
// transposes to MLX format before loading.
// ---------------------------------------------------------------------------

final class _GDNetConv1d extends Module {
  _GDNetConv1d(MLXContext ctx,
      {required int convDim, required int kernelSize})
      : weight = MLXArray.zeros(ctx, [convDim, kernelSize, 1]),
        _convDim = convDim,
        _ctx = ctx;

  MLXArray weight; // [convDim, kernelSize, 1]
  final int _convDim;
  final MLXContext _ctx;

  MLXArray call(MLXArray x) =>
      conv1d(_ctx, x, weight, groups: _convDim);

  @override
  Map<String, MLXArray> parameters() => {'weight': weight};

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (weights['weight'] case final w?) {
      weight.dispose();
      weight = w;
    }
  }

  @override
  void dispose() => weight.dispose();
}

// ---------------------------------------------------------------------------
// SwitchGLU — stacked expert weights for sparse MoE dispatch.
// (Identical to the pattern used in qwen3_moe.dart / deepseek_v3.dart.)
// ---------------------------------------------------------------------------

final class _SwitchGLU extends Module {
  _SwitchGLU(MLXContext ctx,
      {required int inputDims,
      required int hiddenDims,
      required int numExperts})
      : _inputDims = inputDims,
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

  MLXArray call(MLXContext ctx, MLXArray x, MLXArray inds, MLXArray scores) {
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
// GatedDeltaNet layer
//
// Hybrid linear-attention SSM. Uses gatedDeltaUpdate (single-step) in a
// sequential loop over the time axis. For generation (s=1) this is a single
// call; for prefill (s>1) it is a sequential loop.
//
// GQA-style: numVHeads may be a multiple of numKHeads. Q/K are expanded to
// numVHeads heads via repeat before the state update.
// ---------------------------------------------------------------------------

final class _GatedDeltaNetLayer extends Module {
  _GatedDeltaNetLayer(MLXContext ctx, Qwen3NextConfig cfg)
      : numKHeads = cfg.linearNumKeyHeads,
        numVHeads = cfg.linearNumValueHeads,
        keyHeadDim = cfg.linearKeyHeadDim,
        valHeadDim = cfg.linearValueHeadDim,
        keyDim = cfg.keyDim,
        valueDim = cfg.valueDim,
        convDim = cfg.convDim,
        conv = _GDNetConv1d(ctx,
            convDim: cfg.convDim,
            kernelSize: cfg.linearConvKernelDim),
        inProjQkvz = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.keyDim * 2 + cfg.valueDim * 2,
            bias: false),
        inProjBa = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.linearNumValueHeads * 2,
            bias: false),
        // Learnable SSM parameters — initialized to defaults; real values loaded
        aLog = MLXArray.zeros(ctx, [cfg.linearNumValueHeads]),
        dtBias = MLXArray.ones(ctx, [cfg.linearNumValueHeads]),
        norm = RMSNorm(ctx,
            dims: cfg.linearValueHeadDim, eps: cfg.rmsNormEps),
        outProj = Linear(ctx,
            inFeatures: cfg.valueDim,
            outFeatures: cfg.hiddenSize,
            bias: false);

  final int numKHeads;
  final int numVHeads;
  final int keyHeadDim;
  final int valHeadDim;
  final int keyDim;
  final int valueDim;
  final int convDim;

  _GDNetConv1d conv;
  Linear inProjQkvz;
  Linear inProjBa;
  MLXArray aLog;   // [numVHeads]
  MLXArray dtBias; // [numVHeads]
  RMSNorm norm;
  Linear outProj;

  MLXArray call(MLXContext ctx, MLXArray x, MambaCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);
    final vhpk = numVHeads ~/ numKHeads; // V-heads per K-head

    // ---------- Project Q/K/V/Z and B/A ----------
    final qkvzOut = inProjQkvz.call(x); // [B, S, 2*keyDim + 2*valueDim]
    final baOut = inProjBa.call(x);     // [B, S, 2*numVHeads]

    // Slice Q, K, V, Z from flat projection output
    final qFlat =
        qkvzOut.slice(start: [0, 0, 0], stop: [b, s, keyDim])
            .reshape([b, s, numKHeads, keyHeadDim]);
    final kFlat =
        qkvzOut.slice(start: [0, 0, keyDim], stop: [b, s, 2 * keyDim])
            .reshape([b, s, numKHeads, keyHeadDim]);
    final vFlat =
        qkvzOut
            .slice(start: [0, 0, 2 * keyDim],
                stop: [b, s, 2 * keyDim + valueDim])
            .reshape([b, s, numVHeads, valHeadDim]);
    final zFlat =
        qkvzOut
            .slice(start: [0, 0, 2 * keyDim + valueDim],
                stop: [b, s, 2 * keyDim + 2 * valueDim])
            .reshape([b, s, numVHeads, valHeadDim]);
    qkvzOut.dispose();

    // Split BA equally: b=[0:numVHeads], a=[numVHeads:2*numVHeads]
    final baParts = baOut.split(2, axis: -1);
    baOut.dispose();
    final bSlice = baParts[0]; // [B, S, numVHeads]
    final aSlice = baParts[1]; // [B, S, numVHeads]

    // ---------- Conv1d with cached state ----------
    // Build conv input: [convState | mixedQKV]
    // mixedQKV = [q, k, v] concatenated → [B, S, convDim]
    final qConv = qFlat.reshape([b, s, keyDim]);
    final kConv = kFlat.reshape([b, s, keyDim]);
    final vConv = vFlat.reshape([b, s, valueDim]);
    final mixedQKV = concatenate(ctx, [qConv, kConv, vConv], axis: -1);
    qConv.dispose();
    kConv.dispose();
    vConv.dispose();

    final MLXArray convInput;
    final kernelSize = conv.weight.shape[1];
    if (cache?.convState != null) {
      convInput = concatenate(ctx, [cache!.convState!, mixedQKV], axis: 1);
    } else {
      // No prior state: pad left with zeros
      final pad = MLXArray.zeros(ctx, [b, kernelSize - 1, convDim]);
      convInput = concatenate(ctx, [pad, mixedQKV], axis: 1);
      pad.dispose();
    }
    mixedQKV.dispose();

    // Update conv state cache
    if (cache != null) {
      final totalLen = convInput.shape[1];
      final keepLen = kernelSize - 1;
      cache.convState =
          convInput.slice(start: [0, totalLen - keepLen, 0],
              stop: [b, totalLen, convDim]);
    }

    // Apply depthwise conv + SiLU
    final convOut = conv.call(convInput).silu(); // [B, S, convDim]
    convInput.dispose();

    // Split conv output into Q, K, V components
    var qOut =
        convOut.slice(start: [0, 0, 0], stop: [b, s, keyDim])
            .reshape([b, s, numKHeads, keyHeadDim]);
    var kOut =
        convOut
            .slice(start: [0, 0, keyDim], stop: [b, s, 2 * keyDim])
            .reshape([b, s, numKHeads, keyHeadDim]);
    final vOut =
        convOut
            .slice(start: [0, 0, 2 * keyDim],
                stop: [b, s, 2 * keyDim + valueDim])
            .reshape([b, s, numVHeads, valHeadDim]);
    convOut.dispose();

    // Scale-free RMSNorm on Q and K
    final invScale = math.pow(keyHeadDim.toDouble(), -0.5).toDouble();
    final invSq = invScale * invScale;
    final qNormed = _rmsNormNoWeight(ctx, qOut, 1e-6);
    qOut.dispose();
    qOut = qNormed * MLXArray.float_(ctx, invSq);
    qNormed.dispose();
    final kNormed = _rmsNormNoWeight(ctx, kOut, 1e-6);
    kOut.dispose();
    kOut = kNormed * MLXArray.float_(ctx, invScale);
    kNormed.dispose();

    // GQA repeat: expand Q/K to numVHeads if needed
    final qFinal = vhpk > 1 ? qOut.repeat(vhpk, axis: 2) : qOut;
    final kFinal = vhpk > 1 ? kOut.repeat(vhpk, axis: 2) : kOut;
    if (vhpk > 1) {
      qOut.dispose();
      kOut.dispose();
    }

    // ---------- Sequential GatedDelta state update ----------
    var state = cache?.ssmState ??
        MLXArray.zeros(ctx, [b, numVHeads, valHeadDim, keyHeadDim]);
    final ys = <MLXArray>[];

    for (int t = 0; t < s; t++) {
      final aT = aSlice
          .slice(start: [0, t, 0], stop: [b, t + 1, numVHeads])
          .reshape([b, numVHeads]);
      final bT = bSlice
          .slice(start: [0, t, 0], stop: [b, t + 1, numVHeads])
          .reshape([b, numVHeads]);

      // alpha = exp(-exp(aLog) * softplus(aT + dtBias))
      final aLogExp = aLog.exp(); // [numVHeads]
      final softIn = aT + dtBias; // [B, numVHeads]
      aT.dispose();
      final soft = _softplus(ctx, softIn);
      softIn.dispose();
      final alpha = (aLogExp * soft * MLXArray.float_(ctx, -1.0)).exp(); // [B, numVHeads]
      aLogExp.dispose();
      soft.dispose();

      final beta = bT.sigmoid(); // [B, numVHeads]
      bT.dispose();

      final qT = qFinal
          .slice(start: [0, t, 0, 0], stop: [b, t + 1, numVHeads, keyHeadDim])
          .reshape([b, numVHeads, keyHeadDim]);
      final kT = kFinal
          .slice(start: [0, t, 0, 0], stop: [b, t + 1, numVHeads, keyHeadDim])
          .reshape([b, numVHeads, keyHeadDim]);
      final vT = vOut
          .slice(start: [0, t, 0, 0], stop: [b, t + 1, numVHeads, valHeadDim])
          .reshape([b, numVHeads, valHeadDim]);

      final newState =
          gatedDeltaUpdate(ctx, state: state, k: kT, v: vT, beta: beta, alpha: alpha);
      kT.dispose();
      vT.dispose();
      beta.dispose();
      alpha.dispose();
      state.dispose();
      state = newState;

      // yT = einsum('bhde,bhe->bhd', state, qT)
      final yT = einsum(ctx, 'bhde,bhe->bhd', [state, qT]); // [B, nv, dv]
      qT.dispose();
      ys.add(yT);
    }

    cache?.ssmState = state;
    if (cache == null) state.dispose();

    bSlice.dispose();
    aSlice.dispose();
    qFinal.dispose();
    kFinal.dispose();
    vOut.dispose();

    final y = stack(ctx, ys, axis: 1); // [B, S, nv, dv]
    for (final yt in ys) {
      yt.dispose();
    }

    // Gated RMSNorm: norm(y) * silu(z)
    final normedY = norm.call(y); // [B, S, nv, dv]
    y.dispose();
    final gated = normedY * zFlat.silu(); // [B, S, nv, dv]
    normedY.dispose();
    zFlat.dispose();
    qFlat.dispose();
    kFlat.dispose();

    final gatedFlat = gated.reshape([b, s, valueDim]);
    gated.dispose();
    final out = outProj.call(gatedFlat);
    gatedFlat.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in conv.parameters().entries) 'conv1d.${e.key}': e.value,
        for (final e in inProjQkvz.parameters().entries)
          'in_proj_qkvz.${e.key}': e.value,
        for (final e in inProjBa.parameters().entries)
          'in_proj_ba.${e.key}': e.value,
        'A_log': aLog,
        'dt_bias': dtBias,
        for (final e in norm.parameters().entries) 'norm.${e.key}': e.value,
        for (final e in outProj.parameters().entries)
          'out_proj.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    conv.loadWeights(_scoped(weights, 'conv1d'));
    inProjQkvz.loadWeights(_scoped(weights, 'in_proj_qkvz'));
    inProjBa.loadWeights(_scoped(weights, 'in_proj_ba'));
    if (weights['A_log'] case final w?) {
      aLog.dispose();
      aLog = w;
    }
    if (weights['dt_bias'] case final w?) {
      dtBias.dispose();
      dtBias = w;
    }
    norm.loadWeights(_scoped(weights, 'norm'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
  }

  @override
  void dispose() {
    conv.dispose();
    inProjQkvz.dispose();
    inProjBa.dispose();
    aLog.dispose();
    dtBias.dispose();
    norm.dispose();
    outProj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention — standard GQA with QK-norm and gated output.
//
// q_proj outputs numHeads * headDim * 2; the second half is the sigmoid gate
// applied after the attention output before o_proj.
// ---------------------------------------------------------------------------

final class _Qwen3NextAttention extends Module {
  _Qwen3NextAttention(MLXContext ctx, Qwen3NextConfig cfg)
      : numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.attnHeadDim,
        scale = 1.0 / math.sqrt(cfg.attnHeadDim.toDouble()),
        qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.attnHeadDim * 2,
            bias: cfg.attentionBias),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.attnHeadDim,
            bias: cfg.attentionBias),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.attnHeadDim,
            bias: cfg.attentionBias),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.attnHeadDim,
            outFeatures: cfg.hiddenSize,
            bias: cfg.attentionBias),
        qNorm = RMSNorm(ctx, dims: cfg.attnHeadDim, eps: cfg.rmsNormEps),
        kNorm = RMSNorm(ctx, dims: cfg.attnHeadDim, eps: cfg.rmsNormEps),
        _rope = initializeRope(
          ctx,
          dims: math.max(
              1,
              (cfg.attnHeadDim * cfg.partialRotaryFactor).round()),
          traditional: false,
          base: cfg.ropeTheta,
          scalingConfig: cfg.ropeScaling,
          maxPositionEmbeddings: cfg.maxPositionEmbeddings,
        );

  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  RMSNorm qNorm;
  RMSNorm kNorm;
  final RopeLayer _rope;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    // Q projection → split into queries + gate
    final qFull = qProj.call(x)
        .reshape([b, s, numHeads, headDim * 2]);
    final qParts = qFull.split(2, axis: -1);
    qFull.dispose();
    final qRaw = qParts[0]; // [B, S, numHeads, headDim]
    final gate = qParts[1].reshape([b, s, numHeads * headDim]); // [B, S, numHeads*headDim]
    qParts[1].dispose();

    // QK-norm + transpose for RoPE
    final qNormed = qNorm.call(qRaw);
    qRaw.dispose();
    var q = qNormed.transpose([0, 2, 1, 3]); // [B, numHeads, S, headDim]
    qNormed.dispose();

    final kRaw = kProj.call(x).reshape([b, s, numKVHeads, headDim]);
    final kNormed = kNorm.call(kRaw);
    kRaw.dispose();
    var k = kNormed.transpose([0, 2, 1, 3]); // [B, numKVHeads, S, headDim]
    kNormed.dispose();

    final v = vProj.call(x)
        .reshape([b, s, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]); // [B, numKVHeads, S, headDim]

    final offset = cache?.offset ?? 0;
    final qRoped = _rope.call(q, offset);
    q.dispose();
    q = qRoped;
    final kRoped = _rope.call(k, offset);
    k.dispose();
    k = kRoped;

    MLXArray? mask;
    if (s > 1) {
      mask = createCausalMask(ctx, n: s, offset: offset);
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
      maskMode: mask != null ? 'causal' : 'none',
      mask: mask,
    );
    mask?.dispose();
    q.dispose();
    if (cache == null) {
      k.dispose();
      v.dispose();
    }

    // Transpose back and apply sigmoid gate
    final merged = attnOut
        .transpose([0, 2, 1, 3])
        .reshape([b, s, numHeads * headDim]); // [B, S, numHeads*headDim]
    attnOut.dispose();

    final gatedOut = merged * gate.sigmoid();
    merged.dispose();
    gate.dispose();

    final out = oProj.call(gatedOut);
    gatedOut.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in qProj.parameters().entries) 'q_proj.${e.key}': e.value,
        for (final e in kProj.parameters().entries) 'k_proj.${e.key}': e.value,
        for (final e in vProj.parameters().entries) 'v_proj.${e.key}': e.value,
        for (final e in oProj.parameters().entries) 'o_proj.${e.key}': e.value,
        for (final e in qNorm.parameters().entries)
          'q_norm.${e.key}': e.value,
        for (final e in kNorm.parameters().entries)
          'k_norm.${e.key}': e.value,
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
// Dense MLP
// ---------------------------------------------------------------------------

final class _Qwen3NextMLP extends Module {
  _Qwen3NextMLP(MLXContext ctx, int hidden, int intermediate)
      : gateProj = Linear(ctx,
            inFeatures: hidden, outFeatures: intermediate, bias: false),
        downProj = Linear(ctx,
            inFeatures: intermediate, outFeatures: hidden, bias: false),
        upProj = Linear(ctx,
            inFeatures: hidden, outFeatures: intermediate, bias: false);

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
// Sparse MoE block — router + SwitchGLU + shared expert
// ---------------------------------------------------------------------------

final class _Qwen3NextMoEBlock extends Module {
  _Qwen3NextMoEBlock(MLXContext ctx, Qwen3NextConfig cfg)
      : gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numExperts,
            bias: false),
        switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.numExperts),
        sharedExpert = _Qwen3NextMLP(
            ctx, cfg.hiddenSize, cfg.sharedExpertIntermediateSize),
        sharedExpertGate = Linear(ctx,
            inFeatures: cfg.hiddenSize, outFeatures: 1, bias: false),
        _topK = cfg.numExpertsPerTok,
        _normTopkProb = cfg.normTopkProb;

  Linear gate;
  _SwitchGLU switchMlp;
  _Qwen3NextMLP sharedExpert;
  Linear sharedExpertGate;
  final int _topK;
  final bool _normTopkProb;

  MLXArray call(MLXContext ctx, MLXArray x) {
    final logits = gate.call(x); // [b, s, E]
    final scores = logits.softmax(axis: -1);
    logits.dispose();

    final b = x.dim(0);
    final s = x.dim(1);

    // Top-k via argsort (descending)
    final negScores = scores * MLXArray.float_(ctx, -1.0);
    final sortedInds = negScores.argsort(axis: -1);
    negScores.dispose();
    final inds =
        sortedInds.slice(start: [0, 0, 0], stop: [b, s, _topK]); // [b, s, k]
    sortedInds.dispose();

    // Gather routing scores for selected experts
    final e = scores.dim(2);
    final k = _topK;
    final bl = b * s;
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

    var selectedScores =
        scoresFlat2.take(flatInds, axis: 0).reshape([b, s, k]);
    scoresFlat2.dispose();
    flatInds.dispose();

    if (_normTopkProb) {
      final sumS = selectedScores.sum(axis: -1, keepdims: true);
      final norm = selectedScores / sumS;
      sumS.dispose();
      selectedScores.dispose();
      selectedScores = norm;
    }

    final moeOut = switchMlp.call(ctx, x, inds, selectedScores);
    inds.dispose();
    selectedScores.dispose();

    // Shared expert with gating
    final sharedOut = sharedExpert.call(x);
    final sharedGate = sharedExpertGate.call(x).sigmoid();
    final gatedShared = sharedGate * sharedOut;
    sharedOut.dispose();
    sharedGate.dispose();

    final out = moeOut + gatedShared;
    moeOut.dispose();
    gatedShared.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        for (final e in gate.parameters().entries) 'gate.${e.key}': e.value,
        for (final e in switchMlp.parameters().entries)
          'switch_mlp.${e.key}': e.value,
        for (final e in sharedExpert.parameters().entries)
          'shared_expert.${e.key}': e.value,
        for (final e in sharedExpertGate.parameters().entries)
          'shared_expert_gate.${e.key}': e.value,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gate.loadWeights(_scoped(weights, 'gate'));
    switchMlp.loadWeights(_scoped(weights, 'switch_mlp'));
    sharedExpert.loadWeights(_scoped(weights, 'shared_expert'));
    sharedExpertGate.loadWeights(_scoped(weights, 'shared_expert_gate'));
  }

  @override
  void dispose() {
    gate.dispose();
    switchMlp.dispose();
    sharedExpert.dispose();
    sharedExpertGate.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

final class _Qwen3NextDecoderLayer extends Module {
  _Qwen3NextDecoderLayer(MLXContext ctx, Qwen3NextConfig cfg, int layerIdx)
      : isLinear = cfg.isLinearLayer(layerIdx),
        isMoE = cfg.isMoELayer(layerIdx),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps) {
    if (isLinear) {
      linearAttn = _GatedDeltaNetLayer(ctx, cfg);
    } else {
      selfAttn = _Qwen3NextAttention(ctx, cfg);
    }
    if (isMoE) {
      moeBlock = _Qwen3NextMoEBlock(ctx, cfg);
    } else {
      denseMlp = _Qwen3NextMLP(ctx, cfg.hiddenSize, cfg.intermediateSize);
    }
  }

  final bool isLinear;
  final bool isMoE;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;
  _GatedDeltaNetLayer? linearAttn;
  _Qwen3NextAttention? selfAttn;
  _Qwen3NextMoEBlock? moeBlock;
  _Qwen3NextMLP? denseMlp;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final MLXArray h;
    if (isLinear) {
      h = linearAttn!.call(ctx, normed, cache as MambaCache?);
    } else {
      h = selfAttn!.call(ctx, normed, cache);
    }
    normed.dispose();
    final r = x + h;
    h.dispose();

    final postNormed = postAttentionLayernorm.call(r);
    final MLXArray mlpOut;
    if (isMoE) {
      mlpOut = moeBlock!.call(ctx, postNormed);
    } else {
      mlpOut = denseMlp!.call(postNormed);
    }
    postNormed.dispose();

    final out = r + mlpOut;
    mlpOut.dispose();
    r.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in inputLayernorm.parameters().entries)
        'input_layernorm.${e.key}': e.value,
      for (final e in postAttentionLayernorm.parameters().entries)
        'post_attention_layernorm.${e.key}': e.value,
    };
    if (linearAttn != null) {
      for (final e in linearAttn!.parameters().entries) {
        result['linear_attn.${e.key}'] = e.value;
      }
    }
    if (selfAttn != null) {
      for (final e in selfAttn!.parameters().entries) {
        result['self_attn.${e.key}'] = e.value;
      }
    }
    if (moeBlock != null) {
      for (final e in moeBlock!.parameters().entries) {
        result['mlp.${e.key}'] = e.value;
      }
    }
    if (denseMlp != null) {
      for (final e in denseMlp!.parameters().entries) {
        result['mlp.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    linearAttn?.loadWeights(_scoped(weights, 'linear_attn'));
    selfAttn?.loadWeights(_scoped(weights, 'self_attn'));
    moeBlock?.loadWeights(_scoped(weights, 'mlp'));
    denseMlp?.loadWeights(_scoped(weights, 'mlp'));
  }

  @override
  void dispose() {
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
    linearAttn?.dispose();
    selfAttn?.dispose();
    moeBlock?.dispose();
    denseMlp?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner model
// ---------------------------------------------------------------------------

final class _Qwen3NextInnerModel extends Module {
  _Qwen3NextInnerModel(MLXContext ctx, Qwen3NextConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers,
            (i) => _Qwen3NextDecoderLayer(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Qwen3NextDecoderLayer> layers;
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
// Public Qwen3NextModel — implements LanguageModel
//
// Hybrid attention/GatedDeltaNet architecture. Every `fullAttentionInterval`-th
// layer (1-indexed) uses standard attention; the rest use GatedDeltaNet SSM.
// FFN layers are optionally MoE with a shared expert.
//
// Mirrors `Qwen3NextModel` from mlx-swift-lm.
// ---------------------------------------------------------------------------

final class Qwen3NextModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Qwen3NextModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Qwen3NextInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Qwen3NextConfig config;
  final _Qwen3NextInnerModel _model;
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
    return List.generate(config.numHiddenLayers, (i) {
      if (config.isLinearLayer(i)) return MambaCache();
      return KVCacheSimple();
    });
  }

  @override
  List<int> get kvHeads =>
      List.filled(config.numHiddenLayers, config.numKeyValueHeads);

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = <String, MLXArray>{
      for (final e in weights.entries)
          if (!e.key.contains('mtp.')) e.key: e.value,
    };

    if (config.tieWordEmbeddings) {
      result.remove('lm_head.weight');
    }

    // Stack MoE expert weights if not yet stacked
    if (result.containsKey('model.layers.0.mlp.experts.0.up_proj.weight')) {
      for (var l = 0; l < config.numHiddenLayers; l++) {
        final prefix = 'model.layers.$l.mlp';
        for (final n in ['up_proj', 'down_proj', 'gate_proj']) {
          final key = '$prefix.experts.0.$n.weight';
          if (result.containsKey(key)) {
            final tensors = [
              for (var e = 0; e < config.numExperts; e++)
                result.remove('$prefix.experts.$e.$n.weight')!
            ];
            result['$prefix.switch_mlp.$n.weight'] =
                stack(_ctx, tensors, axis: 0);
          }
        }
      }
    }

    // Normalise conv1d weight: PyTorch [C, 1, K] → MLX [C, K, 1]
    // Norm weight offset: HuggingFace stores standard norms as (weight - 1);
    // add 1 here so that RMSNorm uses the correct scale.
    const normSuffixes = [
      '.input_layernorm.weight',
      '.post_attention_layernorm.weight',
      'model.norm.weight',
      '.q_norm.weight',
      '.k_norm.weight',
    ];

    final updated = <String, MLXArray>{};
    for (final e in result.entries) {
      final key = e.key;
      final value = e.value;
      if (key.contains('conv1d.weight') && value.shape.last != 1) {
        // Transpose [C, 1, K] → [C, K, 1]
        updated[key] = value.transpose([0, 2, 1]);
      } else if (normSuffixes.any((s) => key.endsWith(s)) &&
          value.shape.length == 1) {
        updated[key] = value + MLXArray.float_(_ctx, 1.0);
      }
    }
    result.addAll(updated);

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
