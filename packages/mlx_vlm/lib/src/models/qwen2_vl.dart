import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

import '../vision_blocks.dart';
import '../vlm_model.dart';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

final class Qwen2VLVisionConfig {
  const Qwen2VLVisionConfig({
    required this.depth,
    required this.embedDim,
    required this.numHeads,
    required this.hiddenSize,
    this.mlpRatio = 4.0,
    this.inChannels = 3,
    this.patchSize = 14,
    this.spatialMergeSize = 2,
    this.temporalPatchSize = 2,
    this.windowSize = 112,
    this.fullAttBlockIndexes = const [],
  });

  final int depth;
  final int embedDim;
  final int numHeads;
  final int hiddenSize; // language model hidden size (for merger projection)
  final double mlpRatio;
  final int inChannels;
  final int patchSize;
  final int spatialMergeSize;
  final int temporalPatchSize;
  final int windowSize;
  final List<int> fullAttBlockIndexes;

  int get mlpDims => (embedDim * mlpRatio).round();
  int get headDim => embedDim ~/ numHeads;

  factory Qwen2VLVisionConfig.fromJson(Map<String, dynamic> j) =>
      Qwen2VLVisionConfig(
        depth: j['depth'] as int,
        embedDim: j['embed_dim'] as int,
        numHeads: j['num_heads'] as int,
        hiddenSize: j['hidden_size'] as int,
        mlpRatio: (j['mlp_ratio'] as num?)?.toDouble() ?? 4.0,
        inChannels: j['in_channels'] as int? ?? 3,
        patchSize: j['patch_size'] as int? ?? 14,
        spatialMergeSize: j['spatial_merge_size'] as int? ?? 2,
        temporalPatchSize: j['temporal_patch_size'] as int? ?? 2,
        windowSize: j['window_size'] as int? ?? 112,
        fullAttBlockIndexes: (j['fullatt_block_indexes'] as List<dynamic>?)
                ?.cast<int>() ??
            const [],
      );
}

final class Qwen2VLConfig {
  const Qwen2VLConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.vocabSize,
    required this.vision,
    this.rmsNormEps = 1e-6,
    this.ropeTheta = 1000000.0,
    this.mropeSections = const [16, 24, 24],
    this.tieWordEmbeddings = false,
  });

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int vocabSize;
  final Qwen2VLVisionConfig vision;
  final double rmsNormEps;
  final double ropeTheta;
  final List<int> mropeSections; // dims for [t, h, w] RoPE sections
  final bool tieWordEmbeddings;

  int get headDim => hiddenSize ~/ numAttentionHeads;

  factory Qwen2VLConfig.fromJson(Map<String, dynamic> j) {
    final ropeScaling = j['rope_scaling'] as Map<String, dynamic>?;
    final sections = (ropeScaling?['mrope_section'] as List<dynamic>?)
            ?.cast<int>() ??
        const [16, 24, 24];
    return Qwen2VLConfig(
      hiddenSize: j['hidden_size'] as int,
      numHiddenLayers: j['num_hidden_layers'] as int,
      intermediateSize: j['intermediate_size'] as int,
      numAttentionHeads: j['num_attention_heads'] as int,
      numKeyValueHeads: j['num_key_value_heads'] as int? ??
          j['num_attention_heads'] as int,
      vocabSize: j['vocab_size'] as int,
      vision: Qwen2VLVisionConfig.fromJson(
          j['vision_config'] as Map<String, dynamic>),
      rmsNormEps: (j['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
      ropeTheta: (j['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
      mropeSections: sections,
      tieWordEmbeddings: j['tie_word_embeddings'] as bool? ?? false,
    );
  }
}

// ---------------------------------------------------------------------------
// 2-D rotary position embeddings for vision
// ---------------------------------------------------------------------------

/// Pre-compute cos/sin tables for 2-D RoPE given patch grid positions.
///
/// Returns `(cos, sin)` each of shape `[numPatches, headDim]`.
(MLXArray, MLXArray) _computeVisionRoPE(
  MLXContext ctx, {
  required int gridH,
  required int gridW,
  required int headDim,
  double theta = 10000.0,
}) {
  final halfDim = headDim ~/ 4; // per-axis (h and w each get headDim/2 total)

  // Inverse frequencies: [halfDim]
  final iArr = MLXArray.arange(ctx, 0, halfDim.toDouble(), 1.0);
  final thetaArr = MLXArray.float_(ctx, theta);
  final exp = iArr * MLXArray.float_(ctx, 2.0 / headDim.toDouble());
  iArr.dispose();
  final thetaPow = thetaArr.exp(); // workaround: use log/exp for pow
  thetaArr.dispose();
  // inv_freq = 1 / (theta ^ (2i/headDim))
  // Use: theta^x = exp(x * log(theta))
  final logTheta = MLXArray.float_(ctx, math.log(theta));
  final invFreqLog = exp * logTheta;
  exp.dispose();
  logTheta.dispose();
  final invFreq = invFreqLog.exp().rsqrt(); // 1/exp(log(theta)*2i/d) = exp(-log(theta)*2i/d)
  invFreqLog.dispose();
  thetaPow.dispose();

  // Grid positions
  final hPos = MLXArray.arange(ctx, 0, gridH.toDouble(), 1.0); // [gridH]
  final wPos = MLXArray.arange(ctx, 0, gridW.toDouble(), 1.0); // [gridW]

  // freqs_h: [gridH, halfDim], freqs_w: [gridW, halfDim]
  final hPosCol = hPos.reshape([gridH, 1]);
  hPos.dispose();
  final invFreqRow = invFreq.reshape([1, halfDim]);

  final freqsH = hPosCol.matmul(invFreqRow); // [gridH, halfDim]
  hPosCol.dispose();

  final wPosCol = wPos.reshape([gridW, 1]);
  wPos.dispose();
  final freqsW = wPosCol.matmul(invFreqRow); // [gridW, halfDim]
  wPosCol.dispose();
  invFreqRow.dispose();
  invFreq.dispose();

  // Expand to [gridH * gridW, halfDim] by tiling
  final freqsHExp =
      freqsH.reshape([gridH, 1, halfDim]).repeat(gridW, axis: 1).reshape([gridH * gridW, halfDim]);
  freqsH.dispose();
  final freqsWExp =
      freqsW.reshape([1, gridW, halfDim]).repeat(gridH, axis: 0).reshape([gridH * gridW, halfDim]);
  freqsW.dispose();

  // cat along last axis: [numPatches, halfDim*2] = [numPatches, headDim/2]
  final freqs = concatenate(ctx, [freqsHExp, freqsWExp], axis: -1); // [p, headDim/2]
  freqsHExp.dispose();
  freqsWExp.dispose();

  // Duplicate for full head dim: [numPatches, headDim]
  final emb = concatenate(ctx, [freqs, freqs], axis: -1);
  freqs.dispose();

  final cosEmb = emb.cos();
  final sinEmb = emb.sin();
  emb.dispose();
  return (cosEmb, sinEmb);
}

/// Apply rotary position embedding to [x] of shape `[B, numHeads, L, headDim]`.
MLXArray _applyRoPE(MLXContext ctx, MLXArray x, MLXArray cos, MLXArray sin) {
  final parts = x.split(2, axis: -1); // [x1, x2] each [B, H, L, headDim/2]
  final x1 = parts[0];
  final x2 = parts[1];

  // rotate_half: [-x2, x1]
  final negX2 = MLXArray.zeros(ctx, [1]) - x2;
  x2.dispose();
  final xRot = concatenate(ctx, [negX2, x1], axis: -1);
  negX2.dispose();
  x1.dispose();

  // x*cos + rotate_half(x)*sin
  final result = x * cos + xRot * sin;
  xRot.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// Vision attention (fused QKV, 2-D RoPE)
// ---------------------------------------------------------------------------

final class _Qwen2VLVisionAttention extends Module {
  _Qwen2VLVisionAttention(MLXContext ctx, Qwen2VLVisionConfig cfg)
      : qkv = Linear(ctx,
            inFeatures: cfg.embedDim,
            outFeatures: 3 * cfg.embedDim,
            bias: true),
        proj = Linear(ctx,
            inFeatures: cfg.embedDim,
            outFeatures: cfg.embedDim,
            bias: false),
        numHeads = cfg.numHeads,
        headDim = cfg.headDim;

  Linear qkv;
  Linear proj;
  final int numHeads;
  final int headDim;

  MLXArray call(
    MLXContext ctx,
    MLXArray x, // [B, L, embedDim]
    MLXArray cos,
    MLXArray sin,
  ) {
    final b = x.dim(0);
    final s = x.dim(1);

    final qkvOut = qkv.call(x); // [B, L, 3*embedDim]
    final parts = qkvOut.split(3, axis: -1);
    qkvOut.dispose();
    var q = parts[0].reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k = parts[1].reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = parts[2].reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    parts[0].dispose();
    parts[1].dispose();
    parts[2].dispose();

    // Apply 2-D RoPE. cos/sin: [L, headDim] → expand to [1, 1, L, headDim]
    final cosExp = cos.reshape([1, 1, s, headDim]);
    final sinExp = sin.reshape([1, 1, s, headDim]);
    q = _applyRoPE(ctx, q, cosExp, sinExp);
    k = _applyRoPE(ctx, k, cosExp, sinExp);
    cosExp.dispose();
    sinExp.dispose();

    final scale = 1.0 / math.sqrt(headDim.toDouble());
    final attnOut = scaledDotProductAttention(ctx,
        queries: q, keys: k, values: v, scale: scale);
    q.dispose();

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, s, numHeads * headDim]);
    attnOut.dispose();
    final out = proj.call(merged);
    merged.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...qkv.parameters().map((k, v) => MapEntry('qkv.$k', v)),
        ...proj.parameters().map((k, v) => MapEntry('proj.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qkv.loadWeights(_scopedW(weights, 'qkv'));
    proj.loadWeights(_scopedW(weights, 'proj'));
  }

  @override
  void dispose() {
    qkv.dispose();
    proj.dispose();
  }
}

// ---------------------------------------------------------------------------
// Vision encoder block
// ---------------------------------------------------------------------------

final class _Qwen2VLVisionBlock extends Module {
  _Qwen2VLVisionBlock(MLXContext ctx, Qwen2VLVisionConfig cfg)
      : norm1 = LayerNorm(ctx, dims: cfg.embedDim, eps: 1e-6),
        norm2 = LayerNorm(ctx, dims: cfg.embedDim, eps: 1e-6),
        attn = _Qwen2VLVisionAttention(ctx, cfg),
        mlp = VitMLP(ctx, dims: cfg.embedDim, hiddenDims: cfg.mlpDims);

  LayerNorm norm1;
  LayerNorm norm2;
  _Qwen2VLVisionAttention attn;
  VitMLP mlp;

  MLXArray call(
    MLXContext ctx,
    MLXArray x,
    MLXArray cos,
    MLXArray sin,
  ) {
    final n1 = norm1.call(x);
    final a = attn.call(ctx, n1, cos, sin);
    n1.dispose();
    final h = x + a;
    a.dispose();
    final n2 = norm2.call(h);
    final m = mlp.call(n2);
    n2.dispose();
    final out = h + m;
    m.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...norm1.parameters().map((k, v) => MapEntry('norm1.$k', v)),
        ...norm2.parameters().map((k, v) => MapEntry('norm2.$k', v)),
        ...attn.parameters().map((k, v) => MapEntry('attn.$k', v)),
        ...mlp.parameters().map((k, v) => MapEntry('mlp.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    norm1.loadWeights(_scopedW(weights, 'norm1'));
    norm2.loadWeights(_scopedW(weights, 'norm2'));
    attn.loadWeights(_scopedW(weights, 'attn'));
    mlp.loadWeights(_scopedW(weights, 'mlp'));
  }

  @override
  void dispose() {
    norm1.dispose();
    norm2.dispose();
    attn.dispose();
    mlp.dispose();
  }
}

// ---------------------------------------------------------------------------
// Patch merging (spatial compression)
// ---------------------------------------------------------------------------

final class _Qwen2VLMerger extends Module {
  _Qwen2VLMerger(MLXContext ctx, Qwen2VLVisionConfig cfg)
      : ln = LayerNorm(ctx,
            dims: cfg.embedDim * cfg.spatialMergeSize * cfg.spatialMergeSize,
            eps: 1e-6),
        mlp1 = Linear(ctx,
            inFeatures:
                cfg.embedDim * cfg.spatialMergeSize * cfg.spatialMergeSize,
            outFeatures: cfg.hiddenSize),
        mlp2 = Linear(ctx,
            inFeatures: cfg.hiddenSize, outFeatures: cfg.hiddenSize),
        mergeSize = cfg.spatialMergeSize;

  LayerNorm ln;
  Linear mlp1;
  Linear mlp2;
  final int mergeSize;

  /// [x]: `[numPatches, embedDim]`, [thw]: the THW of the grid.
  MLXArray call(MLXContext ctx, MLXArray x, THW thw) {
    final t = thw.t;
    final h = thw.h;
    final w = thw.w;
    final d = x.dim(-1);
    final m = mergeSize;

    // Reshape to [T, H/m, m, W/m, m, D] then merge spatial dims.
    final hm = h ~/ m;
    final wm = w ~/ m;

    // x: [T*H*W, D] → [T, H, W, D]
    var out = x.reshape([t, h, w, d]);

    // Rearrange to [T, H/m, W/m, m*m*D] for spatial merge.
    // Reshape: [T, H/m, m, W/m, m, D]
    out = out.reshape([t, hm, m, wm, m, d]);
    // Transpose to [T, H/m, W/m, m, m, D]
    out = out.transpose([0, 1, 3, 2, 4, 5]);
    // Reshape to [T * H/m * W/m, m*m*D]
    out = out.reshape([t * hm * wm, m * m * d]);

    final normed = ln.call(out);
    out.dispose();
    final h1 = mlp1.call(normed).gelu();
    normed.dispose();
    final h2 = mlp2.call(h1);
    h1.dispose();
    return h2;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...ln.parameters().map((k, v) => MapEntry('ln_q.$k', v)),
        ...mlp1.parameters().map((k, v) => MapEntry('mlp.0.$k', v)),
        ...mlp2.parameters().map((k, v) => MapEntry('mlp.2.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    ln.loadWeights(_scopedW(weights, 'ln_q'));
    mlp1.loadWeights(_scopedW(weights, 'mlp.0'));
    mlp2.loadWeights(_scopedW(weights, 'mlp.2'));
  }

  @override
  void dispose() {
    ln.dispose();
    mlp1.dispose();
    mlp2.dispose();
  }
}

// ---------------------------------------------------------------------------
// Full vision transformer
// ---------------------------------------------------------------------------

final class _Qwen2VLVisionTransformer extends Module implements VisionModel {
  _Qwen2VLVisionTransformer(MLXContext ctx, Qwen2VLVisionConfig cfg)
      : patchEmbed = Linear(
          ctx,
          inFeatures: cfg.inChannels *
              cfg.temporalPatchSize *
              cfg.patchSize *
              cfg.patchSize,
          outFeatures: cfg.embedDim,
          bias: false,
        ),
        blocks = List.generate(cfg.depth, (_) => _Qwen2VLVisionBlock(ctx, cfg)),
        merger = _Qwen2VLMerger(ctx, cfg),
        _cfg = cfg,
        _ctx = ctx;

  final Linear patchEmbed;
  final List<_Qwen2VLVisionBlock> blocks;
  final _Qwen2VLMerger merger;
  final Qwen2VLVisionConfig _cfg;
  final MLXContext _ctx;

  @override
  MLXArray encodeImage(MLXContext ctx, MLXArray pixels) =>
      _forwardWithThw(pixels, const THW(1, 1, 1));

  /// [pixels]: `[numPatches, inC*temporal*pH*pW]`, [thw]: grid dimensions.
  MLXArray forwardWithThw(MLXArray pixels, THW thw) =>
      _forwardWithThw(pixels, thw);

  MLXArray _forwardWithThw(MLXArray pixels, THW thw) {
    // Patch embed: [numPatches, embedDim]
    var x = patchEmbed.call(pixels);
    // Add batch dim for attention: [1, numPatches, embedDim]
    x = x.expandDims(0);

    final gridH = thw.h;
    final gridW = thw.w;

    // Pre-compute 2-D RoPE tables.
    final (cos, sin) =
        _computeVisionRoPE(_ctx, gridH: gridH, gridW: gridW, headDim: _cfg.headDim);

    for (final block in blocks) {
      final next = block.call(_ctx, x, cos, sin);
      x.dispose();
      x = next;
    }
    cos.dispose();
    sin.dispose();

    // Remove batch dim: [numPatches, embedDim]
    final squeezed = x.squeeze(axis: 0);
    x.dispose();

    return merger.call(_ctx, squeezed, thw);
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      ..._scopedMap(patchEmbed.parameters(), 'patch_embed'),
    };
    for (var i = 0; i < blocks.length; i++) {
      for (final e in blocks[i].parameters().entries) {
        result['blocks.$i.${e.key}'] = e.value;
      }
    }
    result.addAll(_scopedMap(merger.parameters(), 'merger'));
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    patchEmbed.loadWeights(_scopedW(weights, 'patch_embed'));
    for (var i = 0; i < blocks.length; i++) {
      blocks[i].loadWeights(_scopedW(weights, 'blocks.$i'));
    }
    merger.loadWeights(_scopedW(weights, 'merger'));
  }

  @override
  void dispose() {
    patchEmbed.dispose();
    for (final b in blocks) {
      b.dispose();
    }
    merger.dispose();
  }
}

// ---------------------------------------------------------------------------
// Language model (Qwen2 = Llama-like with MRoPE)
// ---------------------------------------------------------------------------

// Qwen2's text backbone shares the Llama architecture (GQA, SwiGLU, RMSNorm).
// The primary difference is MRoPE — multimodal rotary PE that uses separate
// position dimensions for temporal/height/width.  For pure-text steps the
// three position components are identical (degrades to standard RoPE).

final class _Qwen2Attention extends Module {
  _Qwen2Attention(MLXContext ctx, Qwen2VLConfig cfg)
      : qProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numAttentionHeads * cfg.headDim),
        kProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim),
        vProj = Linear(ctx,
            inFeatures: cfg.hiddenSize,
            outFeatures: cfg.numKeyValueHeads * cfg.headDim),
        oProj = Linear(ctx,
            inFeatures: cfg.numAttentionHeads * cfg.headDim,
            outFeatures: cfg.hiddenSize),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKeyValueHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        ropeTheta = cfg.ropeTheta,
        mropeSections = cfg.mropeSections;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final double ropeTheta;
  final List<int> mropeSections;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final b = x.dim(0);
    final s = x.dim(1);

    var q = qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    var k = kProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = vProj.call(x).reshape([b, s, numKVHeads, headDim]).transpose([0, 2, 1, 3]);

    final offset = cache?.offset ?? 0;

    // MRoPE: apply standard 1-D RoPE to each mrope section separately.
    // For pure-text or when position_ids aren't provided, all sections use
    // the same sequential positions (equivalent to standard RoPE).
    q = _applyMRoPE(ctx, q, offset: offset, sections: mropeSections, theta: ropeTheta);
    k = _applyMRoPE(ctx, k, offset: offset, sections: mropeSections, theta: ropeTheta);

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

    final attnOut = scaledDotProductAttention(ctx,
        queries: q,
        keys: fullK,
        values: fullV,
        scale: scale,
        maskMode: mask != null ? 'array' : 'none',
        mask: mask);
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
        ...qProj.parameters().map((k, v) => MapEntry('q_proj.$k', v)),
        ...kProj.parameters().map((k, v) => MapEntry('k_proj.$k', v)),
        ...vProj.parameters().map((k, v) => MapEntry('v_proj.$k', v)),
        ...oProj.parameters().map((k, v) => MapEntry('o_proj.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scopedW(weights, 'q_proj'));
    kProj.loadWeights(_scopedW(weights, 'k_proj'));
    vProj.loadWeights(_scopedW(weights, 'v_proj'));
    oProj.loadWeights(_scopedW(weights, 'o_proj'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    oProj.dispose();
  }
}

/// Apply MRoPE using sequential 1-D positions for each section.
///
/// Splits the head dimension into [sections.length] parts and applies
/// independent 1-D RoPE with [offset] to each.  For pure text this
/// degrades to standard RoPE.
MLXArray _applyMRoPE(
  MLXContext ctx,
  MLXArray x, // [B, heads, S, headDim]
  {
  required int offset,
  required List<int> sections,
  required double theta,
}) {
  if (sections.isEmpty) {
    return x.rope(dims: x.dim(-1), base: theta, offset: offset);
  }
  final parts = x.split(sections.length, axis: -1);
  final rotated = <MLXArray>[];
  for (var i = 0; i < parts.length; i++) {
    rotated.add(parts[i].rope(dims: parts[i].dim(-1), base: theta, offset: offset));
    parts[i].dispose();
  }
  final out = concatenate(ctx, rotated, axis: -1);
  for (final r in rotated) {
    r.dispose();
  }
  return out;
}

/// Additive causal mask  (True→0, False→-1e9) shape [n, offset+n].
MLXArray _additiveCausalMask(
  MLXContext ctx, {
  required int n,
  required int offset,
  required MLXDtype dtype,
}) {
  final boolMask = createCausalMask(ctx, n: n, offset: offset);
  final zeros = MLXArray.zeros(ctx, [1], dtype: dtype);
  final neg = MLXArray.fromFloats(ctx, [-1e9]).astype(dtype);
  final result = where(ctx, boolMask, zeros, neg);
  boolMask.dispose();
  zeros.dispose();
  neg.dispose();
  return result;
}

final class _Qwen2MLP extends Module {
  _Qwen2MLP(MLXContext ctx, Qwen2VLConfig cfg)
      : gateProj = Linear(ctx,
            inFeatures: cfg.hiddenSize, outFeatures: cfg.intermediateSize),
        upProj = Linear(ctx,
            inFeatures: cfg.hiddenSize, outFeatures: cfg.intermediateSize),
        downProj = Linear(ctx,
            inFeatures: cfg.intermediateSize, outFeatures: cfg.hiddenSize);

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
    gateProj.loadWeights(_scopedW(weights, 'gate_proj'));
    upProj.loadWeights(_scopedW(weights, 'up_proj'));
    downProj.loadWeights(_scopedW(weights, 'down_proj'));
  }

  @override
  void dispose() {
    gateProj.dispose();
    upProj.dispose();
    downProj.dispose();
  }
}

final class _Qwen2DecoderLayer extends Module {
  _Qwen2DecoderLayer(MLXContext ctx, Qwen2VLConfig cfg)
      : selfAttn = _Qwen2Attention(ctx, cfg),
        mlp = _Qwen2MLP(ctx, cfg),
        inputLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps),
        postLayernorm =
            RMSNorm(ctx, dims: cfg.hiddenSize, eps: cfg.rmsNormEps);

  _Qwen2Attention selfAttn;
  _Qwen2MLP mlp;
  RMSNorm inputLayernorm;
  RMSNorm postLayernorm;

  MLXArray call(MLXContext ctx, MLXArray x, KVCache? cache) {
    final n1 = inputLayernorm.call(x);
    final a = selfAttn.call(ctx, n1, cache);
    n1.dispose();
    final h = x + a;
    a.dispose();
    final n2 = postLayernorm.call(h);
    final m = mlp.call(n2);
    n2.dispose();
    final out = h + m;
    m.dispose();
    h.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...selfAttn.parameters().map((k, v) => MapEntry('self_attn.$k', v)),
        ...mlp.parameters().map((k, v) => MapEntry('mlp.$k', v)),
        ...inputLayernorm.parameters().map((k, v) => MapEntry('input_layernorm.$k', v)),
        ...postLayernorm.parameters().map((k, v) => MapEntry('post_attention_layernorm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    selfAttn.loadWeights(_scopedW(weights, 'self_attn'));
    mlp.loadWeights(_scopedW(weights, 'mlp'));
    inputLayernorm.loadWeights(_scopedW(weights, 'input_layernorm'));
    postLayernorm.loadWeights(_scopedW(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    selfAttn.dispose();
    mlp.dispose();
    inputLayernorm.dispose();
    postLayernorm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public Qwen2VLModel
// ---------------------------------------------------------------------------

/// Qwen2-VL vision language model.
///
/// Architecture: Qwen2VL vision transformer (ViT with 2-D RoPE + spatial
/// merge) combined with a Qwen2 language model backbone (MRoPE).
///
/// Weight key layout mirrors the HuggingFace Qwen2VL checkpoint:
/// - `visual.*`     → vision transformer
/// - `model.*`      → language model (embed_tokens, layers, norm)
/// - `lm_head.*`    → language model head
final class Qwen2VLModel extends Module
    implements VLMModel, KVCacheDimensionProvider {
  Qwen2VLModel(MLXContext ctx, this.config)
      : _ctx = ctx,
        _visual = _Qwen2VLVisionTransformer(ctx, config.vision),
        _embedTokens =
            Embedding(ctx, vocabSize: config.vocabSize, dims: config.hiddenSize),
        _layers = List.generate(
            config.numHiddenLayers, (_) => _Qwen2DecoderLayer(ctx, config)),
        _norm = RMSNorm(ctx, dims: config.hiddenSize, eps: config.rmsNormEps),
        _lmHead = Linear(ctx,
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            bias: false);

  final MLXContext _ctx;
  final Qwen2VLConfig config;
  final _Qwen2VLVisionTransformer _visual;
  final Embedding _embedTokens;
  final List<_Qwen2DecoderLayer> _layers;
  final RMSNorm _norm;
  final Linear _lmHead;

  // -------------------------------------------------------------------------
  // VLMModel
  // -------------------------------------------------------------------------

  @override
  VisionModel get visionModel => _visual;

  @override
  LMOutput callWithPixels(
    MLXContext ctx,
    LMInput input, {
    List<KVCache>? cache,
    LMState? state,
  }) {
    // Embed text tokens.
    var h = _embedTokens.call(input.text.tokens); // [B, S, D]

    // If an image is present, encode it and splice into the embedding sequence
    // at the positions of IMAGE_PAD tokens in the token stream.
    if (input.image case final img?) {
      final thw = (img.frames?.isNotEmpty == true)
          ? img.frames!.first
          : THW(1, _inferGrid(img.pixels), _inferGrid(img.pixels));
      final visualFeats =
          _visual.forwardWithThw(img.pixels, thw); // [numMergedPatches, D]

      // Find the contiguous range of IMAGE_PAD tokens (id = 151655).
      // Evaluate the 1-D token sequence on CPU to locate them.
      const imagePadId = 151655;
      final tokenTensor = input.text.tokens; // [B, S] or [S]
      final flat = tokenTensor.ndim > 1
          ? tokenTensor.reshape([tokenTensor.size])
          : tokenTensor;
      flat.eval();
      _ctx.synchronize();
      final ids = flat.toInt32List();
      flat.dispose();

      int imgStart = -1, imgEnd = -1;
      for (var i = 0; i < ids.length; i++) {
        if (ids[i] == imagePadId) {
          if (imgStart == -1) imgStart = i;
          imgEnd = i + 1;
        }
      }

      if (imgStart >= 0) {
        // Splice: h[0, imgStart:imgEnd, :] = visualFeats  ([numPatches, D])
        final numPatches = imgEnd - imgStart;
        final d = h.dim(2);
        final patch = visualFeats.reshape([1, numPatches, d]);
        final spliced = h.sliceUpdate(patch,
            start: [0, imgStart, 0], stop: [1, imgEnd, d]);
        patch.dispose();
        h.dispose();
        h = spliced;
      }
      visualFeats.dispose();
    }

    for (var i = 0; i < _layers.length; i++) {
      final c = (cache != null && i < cache.length) ? cache[i] : null;
      final next = _layers[i].call(_ctx, h, c);
      h.dispose();
      h = next;
    }
    final normed = _norm.call(h);
    h.dispose();
    final logits = _lmHead.call(normed);
    normed.dispose();
    return LMOutput(logits: logits);
  }

  // -------------------------------------------------------------------------
  // LanguageModel
  // -------------------------------------------------------------------------

  @override
  PrepareResult prepare(LMInput input, List<KVCache> cache, {int? windowSize}) {
    return PrepareResult.logits(
        callWithPixels(_ctx, input, cache: cache));
  }

  @override
  LMOutput call(LMInputText input, {List<KVCache>? cache, LMState? state}) {
    final lmInput = LMInput(text: input);
    return callWithPixels(_ctx, lmInput, cache: cache, state: state);
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
  // Weights
  // -------------------------------------------------------------------------

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    if (!weights.containsKey('lm_head.weight') &&
        weights.containsKey('model.embed_tokens.weight')) {
      return {...weights, 'lm_head.weight': weights['model.embed_tokens.weight']!};
    }
    return weights;
  }

  @override
  Map<String, MLXArray> parameters() {
    final result = <String, MLXArray>{
      ..._scopedMap(_visual.parameters(), 'visual'),
      ..._scopedMap(_embedTokens.parameters(), 'model.embed_tokens'),
      ..._scopedMap(_norm.parameters(), 'model.norm'),
      ..._scopedMap(_lmHead.parameters(), 'lm_head'),
    };
    for (var i = 0; i < _layers.length; i++) {
      for (final e in _layers[i].parameters().entries) {
        result['model.layers.$i.${e.key}'] = e.value;
      }
    }
    return result;
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    _visual.loadWeights(_scopedW(weights, 'visual'));
    _embedTokens.loadWeights(_scopedW(weights, 'model.embed_tokens'));
    for (var i = 0; i < _layers.length; i++) {
      _layers[i].loadWeights(_scopedW(weights, 'model.layers.$i'));
    }
    _norm.loadWeights(_scopedW(weights, 'model.norm'));
    _lmHead.loadWeights(_scopedW(weights, 'lm_head'));
  }

  @override
  void dispose() {
    _visual.dispose();
    _embedTokens.dispose();
    for (final l in _layers) {
      l.dispose();
    }
    _norm.dispose();
    _lmHead.dispose();
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  static int _inferGrid(MLXArray pixels) {
    // pixels: [numPatches, flatPatchDim] — estimate grid from patch count
    final n = pixels.dim(0);
    return math.sqrt(n.toDouble()).round();
  }
}

// ---------------------------------------------------------------------------
// Internal weight-key helpers (file-private)
// ---------------------------------------------------------------------------

Map<String, MLXArray> _scopedW(Map<String, MLXArray> w, String prefix) {
  final p = '$prefix.';
  return {
    for (final e in w.entries)
      if (e.key.startsWith(p)) e.key.substring(p.length): e.value,
  };
}

Map<String, MLXArray> _scopedMap(Map<String, MLXArray> m, String prefix) =>
    m.map((k, v) => MapEntry('$prefix.$k', v));
