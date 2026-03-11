import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Qwen3MoEConfig {
  const Qwen3MoEConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    this.rmsNormEps = 1e-6,
    required this.vocabSize,
    this.ropeTheta = 1000000.0,
    this.ropeScaling,
    this.maxPositionEmbeddings = 32768,
    this.tieWordEmbeddings = false,
    required this.numExperts,
    required this.numExpertsPerToken,
    required this.moeIntermediateSize,
    this.decoderSparseStep = 1,
    this.mlpOnlyLayers = const [],
    this.normTopkProb = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final int maxPositionEmbeddings;
  final bool tieWordEmbeddings;
  final int numExperts;
  final int numExpertsPerToken;
  final int moeIntermediateSize;
  final int decoderSparseStep;
  final List<int> mlpOnlyLayers;
  final bool normTopkProb;

  factory Qwen3MoEConfig.fromJson(Map<String, dynamic> j) => Qwen3MoEConfig(
        hiddenSize: j['hidden_size'] as int,
        numHiddenLayers: j['num_hidden_layers'] as int,
        intermediateSize: j['intermediate_size'] as int,
        numAttentionHeads: j['num_attention_heads'] as int,
        numKeyValueHeads:
            j['num_key_value_heads'] as int? ?? j['num_attention_heads'] as int,
        headDim: j['head_dim'] as int? ??
            (j['hidden_size'] as int) ~/ (j['num_attention_heads'] as int),
        rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
        vocabSize: j['vocab_size'] as int,
        ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
        ropeScaling: j['rope_scaling'] as Map<String, dynamic>?,
        maxPositionEmbeddings:
            j['max_position_embeddings'] as int? ?? 32768,
        tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
        numExperts: j['num_experts'] as int,
        numExpertsPerToken: j['num_experts_per_tok'] as int,
        moeIntermediateSize: j['moe_intermediate_size'] as int,
        decoderSparseStep: j['decoder_sparse_step'] as int? ?? 1,
        mlpOnlyLayers: (j['mlp_only_layers'] as List?)
                ?.map((v) => v as int)
                .toList() ??
            const [],
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
// Dense MLP — used on non-MoE layers
// ---------------------------------------------------------------------------

final class _Qwen3MoEDenseMLP extends Module {
  _Qwen3MoEDenseMLP(MLXContext ctx, int hidden, int intermediate)
      : gateProj = Linear(ctx,
            inFeatures: hidden,
            outFeatures: intermediate,
            bias: false),
        downProj = Linear(ctx,
            inFeatures: intermediate,
            outFeatures: hidden,
            bias: false),
        upProj = Linear(ctx,
            inFeatures: hidden,
            outFeatures: intermediate,
            bias: false);

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
// SwitchGLU — stacked expert weights, full-compute then gather
//
// Weights stored as [numExperts, hiddenDims, inputDims] after sanitize stacks
// individual expert tensors along axis 0.
//
// Forward:
//   1. Project x to all experts at once via reshape trick.
//   2. Apply silu-gated activation.
//   3. Down-project.
//   4. Gather top-k expert outputs.
//   5. Weighted sum.
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
        // Weights will be loaded; initialize as zero placeholders
        gateProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        upProj = MLXArray.zeros(ctx, [numExperts, hiddenDims, inputDims]),
        downProj = MLXArray.zeros(ctx, [numExperts, inputDims, hiddenDims]);

  final int _inputDims;
  final int _hiddenDims;
  final int _numExperts;

  MLXArray gateProj; // [E, h, d]
  MLXArray upProj;   // [E, h, d]
  MLXArray downProj; // [E, d, h]

  /// Compute outputs for ALL experts in a batched fashion.
  ///
  /// Returns `[b, s, E, d]` — per-expert output for every position.
  MLXArray _computeAll(MLXContext ctx, MLXArray x) {
    final b = x.dim(0);
    final s = x.dim(1);
    final d = _inputDims;
    final h = _hiddenDims;
    final e = _numExperts;
    final bl = b * s;

    final xFlat = x.reshape([bl, d]); // [B*L, d]

    // gate_all = x @ gate_proj.reshape([E*h, d]).T → [B*L, E*h] → [B*L, E, h]
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

    // down: [B*L, E, h] → [E, B*L, h] @ [E, h, d] → [E, B*L, d] → [B*L, E, d]
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

  /// Gather top-k expert outputs and apply weighted sum.
  ///
  /// [expertOuts]: `[b, s, E, d]`
  /// [inds]: `[b, s, k]` int32 expert indices
  /// [scores]: `[b, s, k]` routing weights
  /// Returns `[b, s, d]`.
  MLXArray _gatherAndSum(
      MLXContext ctx, MLXArray expertOuts, MLXArray inds, MLXArray scores) {
    final b = expertOuts.dim(0);
    final s = expertOuts.dim(1);
    final e = _numExperts;
    final d = _inputDims;
    final k = inds.dim(2);

    // Flatten to [b*s, E, d] and [b*s, k]
    final outFlat = expertOuts.reshape([b * s, e, d]);
    final indsFlat = inds.reshape([b * s, k]); // [b*s, k]

    // Compute flat indices into [b*s * E, d]: flat_idx[i, j] = i*E + inds[i,j]
    final posRange = MLXArray.arange(ctx, 0.0, (b * s).toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([b * s, 1])
        .repeat(k, axis: 1); // [b*s, k]
    final eArr = MLXArray.int_(ctx, e);
    final offsets = posRange * eArr;
    posRange.dispose();
    eArr.dispose();

    final flatInds = (offsets + indsFlat).reshape([b * s * k]); // [b*s*k]
    offsets.dispose();
    indsFlat.dispose();

    // Flatten outFlat to [b*s*E, d] then take
    final outFlat2 = outFlat.reshape([b * s * e, d]);
    outFlat.dispose();
    final selected = outFlat2.take(flatInds, axis: 0); // [b*s*k, d]
    outFlat2.dispose();
    flatInds.dispose();

    // Reshape to [b, s, k, d] and apply per-expert weights
    final selectedBSKD = selected.reshape([b, s, k, d]);
    selected.dispose();
    final scoresExpanded = scores.expandDims(3); // [b, s, k, 1]
    final weighted = selectedBSKD * scoresExpanded; // [b, s, k, d]
    selectedBSKD.dispose();
    scoresExpanded.dispose();

    final out = weighted.sum(axis: 2); // [b, s, d]
    weighted.dispose();
    return out;
  }

  /// Full MoE dispatch: compute all experts, gather top-k, weighted sum.
  ///
  /// [x]: `[b, s, d]`
  /// [inds]: `[b, s, k]` int32 — selected expert indices
  /// [scores]: `[b, s, k]` float — routing weights
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
// Sparse MoE block — router + SwitchGLU
// ---------------------------------------------------------------------------

final class _SparseMoEBlock extends Module {
  _SparseMoEBlock(MLXContext ctx, Qwen3MoEConfig cfg)
      : gate = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numExperts,
            bias: false),
        switchMlp = _SwitchGLU(ctx,
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.numExperts),
        _topK = cfg.numExpertsPerToken,
        _normTopkProb = cfg.normTopkProb;

  Linear gate;
  _SwitchGLU switchMlp;
  final int _topK;
  final bool _normTopkProb;

  MLXArray call(MLXContext ctx, MLXArray x) {
    // Router logits and softmax scores
    final logits = gate.call(x); // [b, s, E]
    final scores = logits.softmax(axis: -1); // [b, s, E]

    // Top-k expert selection: argsort descending, take first k
    final negLogits = logits * MLXArray.float_(ctx, -1.0);
    logits.dispose();
    final sortedInds = negLogits.argsort(axis: -1); // ascending by -logits = descending by logits
    negLogits.dispose();

    final b = x.dim(0);
    final s = x.dim(1);
    final inds =
        sortedInds.slice(start: [0, 0, 0], stop: [b, s, _topK]); // [b, s, k]
    sortedInds.dispose();

    // Gather routing scores for selected experts
    final e = scores.dim(2);
    final d = _topK;
    final bl = b * s;

    final scoresFlat = scores.reshape([bl, e]);
    final indsFlat = inds.reshape([bl, d]);

    final posRange = MLXArray.arange(ctx, 0.0, bl.toDouble(), 1.0,
            dtype: MLXDtype.int32)
        .reshape([bl, 1])
        .repeat(d, axis: 1); // [bl, k]
    final eArr = MLXArray.int_(ctx, e);
    final offsets = posRange * eArr;
    posRange.dispose();
    eArr.dispose();

    final flatInds = (offsets + indsFlat).reshape([bl * d]);
    offsets.dispose();
    indsFlat.dispose();

    final scoresFlat2 = scoresFlat.reshape([bl * e]);
    scoresFlat.dispose();
    scores.dispose();

    var selectedScores =
        scoresFlat2.take(flatInds, axis: 0).reshape([b, s, d]); // [b, s, k]
    scoresFlat2.dispose();
    flatInds.dispose();

    if (_normTopkProb) {
      final sumScores = selectedScores.sum(axis: -1, keepdims: true); // [b, s, 1]
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
// Attention — with QK-norm
// ---------------------------------------------------------------------------

final class _Qwen3MoEAttention extends Module {
  _Qwen3MoEAttention(MLXContext ctx, Qwen3MoEConfig cfg)
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
        qNorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
        kNorm = RMSNorm(ctx, dims: cfg.headDim, eps: cfg.rmsNormEps),
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

    // Project, reshape, apply QK-norms, then RoPE
    final qRaw = qProj
        .call(x)
        .reshape([b, s, numHeads, headDim]);
    final qNormed = qNorm.call(qRaw);
    qRaw.dispose();
    final q = qNormed.transpose([0, 2, 1, 3]);
    qNormed.dispose();

    final kRaw = kProj
        .call(x)
        .reshape([b, s, numKVHeads, headDim]);
    final kNormed = kNorm.call(kRaw);
    kRaw.dispose();
    final k = kNormed.transpose([0, 2, 1, 3]);
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
// Decoder layer
// ---------------------------------------------------------------------------

final class _Qwen3MoEDecoderLayer extends Module {
  _Qwen3MoEDecoderLayer(MLXContext ctx, Qwen3MoEConfig cfg, int layerIdx)
      : selfAttn = _Qwen3MoEAttention(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postAttentionLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        _isMoE = !cfg.mlpOnlyLayers.contains(layerIdx) &&
            cfg.numExperts > 0 &&
            (layerIdx + 1) % cfg.decoderSparseStep == 0,
        _denseMlp = (!cfg.mlpOnlyLayers.contains(layerIdx) &&
                    cfg.numExperts > 0 &&
                    (layerIdx + 1) % cfg.decoderSparseStep == 0)
                ? null
                : _Qwen3MoEDenseMLP(
                    ctx, cfg.hiddenSize, cfg.intermediateSize),
        _moeMlp = (!cfg.mlpOnlyLayers.contains(layerIdx) &&
                    cfg.numExperts > 0 &&
                    (layerIdx + 1) % cfg.decoderSparseStep == 0)
                ? _SparseMoEBlock(ctx, cfg)
                : null;

  _Qwen3MoEAttention selfAttn;
  RMSNorm inputLayernorm;
  RMSNorm postAttentionLayernorm;
  final bool _isMoE;
  final _Qwen3MoEDenseMLP? _denseMlp;
  final _SparseMoEBlock? _moeMlp;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final normed = inputLayernorm.call(x);
    final attnOut = selfAttn.call(ctx, normed, cache);
    normed.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    final normed2 = postAttentionLayernorm.call(afterAttn);
    final mlpOut = _isMoE
        ? _moeMlp!.call(ctx, normed2)
        : _denseMlp!.call(normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      for (final e in selfAttn.parameters().entries)
        'self_attn.${e.key}': e.value,
      for (final e in inputLayernorm.parameters().entries)
        'input_layernorm.${e.key}': e.value,
      for (final e in postAttentionLayernorm.parameters().entries)
        'post_attention_layernorm.${e.key}': e.value,
    };
    if (_isMoE) {
      for (final e in _moeMlp!.parameters().entries) {
        result['mlp.${e.key}'] = e.value;
      }
    } else {
      for (final e in _denseMlp!.parameters().entries) {
        result['mlp.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scoped(weights, 'self_attn'));
    inputLayernorm.loadWeights(_scoped(weights, 'input_layernorm'));
    postAttentionLayernorm
        .loadWeights(_scoped(weights, 'post_attention_layernorm'));
    if (_isMoE) {
      _moeMlp!.loadWeights(_scoped(weights, 'mlp'));
    } else {
      _denseMlp!.loadWeights(_scoped(weights, 'mlp'));
    }
  }

  @override
  void dispose() {
    selfAttn.dispose();
    inputLayernorm.dispose();
    postAttentionLayernorm.dispose();
    _denseMlp?.dispose();
    _moeMlp?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inner transformer
// ---------------------------------------------------------------------------

final class _Qwen3MoEInnerModel extends Module {
  _Qwen3MoEInnerModel(MLXContext ctx, Qwen3MoEConfig cfg)
      : embedTokens =
            Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenSize),
        layers = List.generate(cfg.numHiddenLayers,
            (i) => _Qwen3MoEDecoderLayer(ctx, cfg, i)),
        norm = RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  Embedding embedTokens;
  List<_Qwen3MoEDecoderLayer> layers;
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
// Public Qwen3MoEModel — implements LanguageModel
// ---------------------------------------------------------------------------

/// Qwen3 Mixture-of-Experts language model.
///
/// Each MoE layer uses a top-k sparse router over stacked expert GLU blocks.
/// Non-MoE layers (controlled by `mlpOnlyLayers` and `decoderSparseStep`) use
/// a standard dense SiLU-gated MLP.
///
/// Mirrors `Qwen3MoEModel` from mlx-swift-lm.
final class Qwen3MoEModel extends Module
    implements LanguageModel, KVCacheDimensionProvider {
  Qwen3MoEModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _model = _Qwen3MoEInnerModel(ctx, config),
        _lmHead = config.tieWordEmbeddings
            ? null
            : Linear(ctx,
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                bias: false);

  final MLXContext _ctx;
  final Qwen3MoEConfig config;
  final _Qwen3MoEInnerModel _model;
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

  /// Stack per-expert weights from `mlp.experts.N.{gate,up,down}_proj.weight`
  /// into `mlp.switch_mlp.{gate,up,down}_proj.weight` shape `[E, h, d]`.
  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final result = Map<String, MLXArray>.of(weights);

    if (config.tieWordEmbeddings) {
      result.remove('lm_head.weight');
    }

    // Check if stacking is needed (individual expert keys exist)
    if (!result.containsKey('model.layers.0.mlp.experts.0.up_proj.weight')) {
      return result;
    }

    for (var l = 0; l < config.numHiddenLayers; l++) {
      final prefix = 'model.layers.$l';
      for (final n in ['up_proj', 'down_proj', 'gate_proj']) {
        final key0 = '$prefix.mlp.experts.0.$n.weight';
        if (result.containsKey(key0)) {
          final toStack = <MLXArray>[];
          for (var ei = 0; ei < config.numExperts; ei++) {
            final k = '$prefix.mlp.experts.$ei.$n.weight';
            toStack.add(result.remove(k)!);
          }
          result['$prefix.mlp.switch_mlp.$n.weight'] =
              stack(toStack.first.context, toStack, axis: 0);
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
