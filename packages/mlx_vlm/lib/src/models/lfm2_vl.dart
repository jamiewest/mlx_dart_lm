import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

import '../vision_blocks.dart';
import '../vlm_model.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// softplus(x) = log(1 + exp(x)), numerically stable for large x.
MLXArray _softplus(MLXContext ctx, MLXArray x) {
  final ex = x.exp();
  final one = MLXArray.fromFloats(ctx, [1.0]).astype(ex.dtype);
  final sum = ex + one;
  ex.dispose();
  one.dispose();
  final result = sum.log();
  sum.dispose();
  return result;
}

Map<String, MLXArray> _scoped(Map<String, MLXArray> w, String prefix) {
  final p = '$prefix.';
  return {
    for (final e in w.entries)
      if (e.key.startsWith(p)) e.key.substring(p.length): e.value,
  };
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

final class Lfm2VisionConfig {
  const Lfm2VisionConfig({
    required this.imageSize,
    required this.patchSize,
    required this.numLayers,
    required this.numHeads,
    required this.hiddenDim,
    required this.mlpDim,
    this.numChannels = 3,
    this.eps = 1e-6,
  });

  final int imageSize;
  final int patchSize;
  final int numLayers;
  final int numHeads;
  final int hiddenDim;
  final int mlpDim;
  final int numChannels;
  final double eps;

  int get numPatches => (imageSize ~/ patchSize) * (imageSize ~/ patchSize);
  int get headDim => hiddenDim ~/ numHeads;

  factory Lfm2VisionConfig.fromJson(Map<String, dynamic> json) {
    return Lfm2VisionConfig(
      imageSize: (json['image_size'] as num?)?.toInt() ?? 384,
      patchSize: (json['patch_size'] as num?)?.toInt() ?? 16,
      numLayers: (json['num_hidden_layers'] as num?)?.toInt() ?? 27,
      numHeads: (json['num_attention_heads'] as num?)?.toInt() ?? 16,
      hiddenDim: (json['hidden_size'] as num?)?.toInt() ?? 1152,
      mlpDim: (json['intermediate_size'] as num?)?.toInt() ?? 4304,
      numChannels: (json['num_channels'] as num?)?.toInt() ?? 3,
      eps: (json['layer_norm_eps'] as num?)?.toDouble() ?? 1e-6,
    );
  }
}

final class Lfm2Config {
  const Lfm2Config({
    required this.vocabSize,
    required this.hiddenDim,
    required this.numLayers,
    required this.numAttentionHeads,
    required this.numKVHeads,
    required this.headDim,
    required this.mlpDim,
    required this.ssmInnerDim,
    required this.ssmDState,
    required this.ssmDtRank,
    required this.ssmConvSize,
    required this.attentionLayerIndices,
    required this.visionConfig,
    this.rmsNormEps = 1e-5,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.projectorHiddenDim,
  });

  final int vocabSize;
  final int hiddenDim;
  final int numLayers;
  final int numAttentionHeads;
  final int numKVHeads;
  final int headDim;
  final int mlpDim;

  // SSM parameters
  final int ssmInnerDim;
  final int ssmDState;
  final int ssmDtRank;
  final int ssmConvSize;

  /// Indices of decoder layers that use attention (rest use SSM).
  final List<int> attentionLayerIndices;

  final Lfm2VisionConfig visionConfig;
  final double rmsNormEps;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;

  /// Hidden dim for the vision _projector MLP. Defaults to [hiddenDim].
  final int? projectorHiddenDim;

  int get effectiveProjectorHiddenDim => projectorHiddenDim ?? hiddenDim;

  factory Lfm2Config.fromJson(Map<String, dynamic> json) {
    final visionJson = (json['vision_config'] as Map<String, dynamic>?) ?? {};
    final attnIndices = (json['attention_layer_indices'] as List<dynamic>?)
            ?.map((e) => (e as num).toInt())
            .toList() ??
        [];

    final hiddenDim = (json['hidden_size'] as num?)?.toInt() ?? 2048;
    final numLayers = (json['num_hidden_layers'] as num?)?.toInt() ?? 28;
    final numHeads = (json['num_attention_heads'] as num?)?.toInt() ?? 16;
    final numKVHeads =
        (json['num_key_value_heads'] as num?)?.toInt() ?? numHeads;
    final headDim =
        (json['head_dim'] as num?)?.toInt() ?? hiddenDim ~/ numHeads;
    final mlpDim = (json['intermediate_size'] as num?)?.toInt() ?? 4 * hiddenDim;
    final ssmInner = (json['ssm_inner_dim'] as num?)?.toInt() ?? 2 * hiddenDim;
    final ssmDState = (json['ssm_d_state'] as num?)?.toInt() ?? 16;
    final ssmDtRank = (json['ssm_dt_rank'] as num?)?.toInt() ??
        (hiddenDim / 16).ceil();
    final ssmConv = (json['ssm_conv_size'] as num?)?.toInt() ?? 4;

    // If no attention_layer_indices key, assume every 7th layer is attention.
    final effectiveAttn = attnIndices.isNotEmpty
        ? attnIndices
        : List.generate(numLayers, (i) => i)
            .where((i) => i % 7 == 1)
            .toList();

    return Lfm2Config(
      vocabSize: (json['vocab_size'] as num?)?.toInt() ?? 128256,
      hiddenDim: hiddenDim,
      numLayers: numLayers,
      numAttentionHeads: numHeads,
      numKVHeads: numKVHeads,
      headDim: headDim,
      mlpDim: mlpDim,
      ssmInnerDim: ssmInner,
      ssmDState: ssmDState,
      ssmDtRank: ssmDtRank,
      ssmConvSize: ssmConv,
      attentionLayerIndices: effectiveAttn,
      visionConfig: Lfm2VisionConfig.fromJson(visionJson),
      rmsNormEps:
          (json['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5,
      ropeTheta:
          (json['rope_theta'] as num?)?.toDouble() ?? 10000.0,
      ropeScaling: json['rope_scaling'] as Map<String, dynamic>?,
      projectorHiddenDim:
          (json['_projector_hidden_dim'] as num?)?.toInt(),
    );
  }
}

// ---------------------------------------------------------------------------
// SsmCache — recurrent state for SSM layers
// ---------------------------------------------------------------------------

/// Cache that holds the recurrent state of a single Mamba SSM layer.
///
/// [convState] shape: `[1, d_inner, conv_size]`
/// [ssmState] shape: `[1, d_inner, d_state]`
final class SsmCache implements KVCache {
  MLXArray? convState;
  MLXArray? ssmState;

  @override
  int get offset => ssmState == null ? 0 : 1;

  @override
  int? get maxSize => null;

  @override
  (MLXArray, MLXArray) update(MLXArray keys, MLXArray values) =>
      throw UnsupportedError('SsmCache does not use KV-style update');

  @override
  List<MLXArray> get state => [
        if (convState != null) convState!,
        if (ssmState != null) ssmState!,
      ];

  @override
  set state(List<MLXArray> value) {
    if (value.length >= 2) {
      convState = value[0];
      ssmState = value[1];
    }
  }

  @override
  bool get isTrimmable => false;

  @override
  int trim(int n) => 0;

  @override
  MLXArray? makeMask(MLXContext ctx, {required int n, int? windowSize}) => null;

  @override
  String get cacheType => 'ssm';

  void dispose() {
    convState?.dispose();
    ssmState?.dispose();
    convState = null;
    ssmState = null;
  }
}

// ---------------------------------------------------------------------------
// SigLIP Vision Encoder
// ---------------------------------------------------------------------------

final class _SigLIPEncoder extends Module implements VisionModel {
  _SigLIPEncoder(MLXContext ctx, Lfm2VisionConfig cfg)
      : patchEmbed = Conv2d(
          ctx,
          inChannels: cfg.numChannels,
          outChannels: cfg.hiddenDim,
          kernelSize: cfg.patchSize,
          stride: cfg.patchSize,
          padding: 0,
          bias: true,
        ),
        posEmbed = MLXArray.zeros(ctx, [1, cfg.numPatches, cfg.hiddenDim]),
        blocks = List.generate(
          cfg.numLayers,
          (_) => VitEncoderBlock(
            ctx,
            dims: cfg.hiddenDim,
            numHeads: cfg.numHeads,
            mlpDims: cfg.mlpDim,
            eps: cfg.eps,
            attentionBias: true,
            useGelu: true,
          ),
        ),
        norm = LayerNorm(ctx, dims: cfg.hiddenDim, eps: cfg.eps, bias: true),
        _ctx = ctx;

  Conv2d patchEmbed;
  MLXArray posEmbed;
  final List<VitEncoderBlock> blocks;
  LayerNorm norm;
  final MLXContext _ctx;

  @override
  MLXArray encodeImage(MLXContext ctx, MLXArray pixels) {
    // pixels: [B, H, W, C]  (NHWC, float32, range [-1,1])
    final b = pixels.dim(0);
    final h = pixels.dim(1);
    final w = pixels.dim(2);

    // Patch embed: [B, H', W', embedDim]
    final embedded = patchEmbed.call(pixels);
    final hp = embedded.dim(1);
    final wp = embedded.dim(2);

    // Reshape to [B, numPatches, embedDim] and add position embedding.
    final flat = embedded.reshape([b, hp * wp, embedded.dim(3)]);
    embedded.dispose();
    var x = flat + posEmbed;
    flat.dispose();

    for (final block in blocks) {
      final next = block.call(_ctx, x);
      x.dispose();
      x = next;
    }

    final normed = norm.call(x);
    x.dispose();
    // Suppress unused-variable warnings for h/w — they are used implicitly via
    // the conv stride but not explicitly in Dart. Keep as assertions if needed.
    assert(h > 0 && w > 0);
    return normed; // [B, numPatches, hiddenDim]
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...patchEmbed.parameters().map((k, v) => MapEntry('patch_embed.$k', v)),
        'pos_embed': posEmbed,
        for (var i = 0; i < blocks.length; i++)
          ...blocks[i].parameters().map((k, v) => MapEntry('blocks.$i.$k', v)),
        ...norm.parameters().map((k, v) => MapEntry('norm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    patchEmbed.loadWeights(_scoped(weights, 'patch_embed'));
    if (weights['pos_embed'] case final pe?) posEmbed = pe;
    for (var i = 0; i < blocks.length; i++) {
      blocks[i].loadWeights(_scoped(weights, 'blocks.$i'));
    }
    norm.loadWeights(_scoped(weights, 'norm'));
  }

  @override
  void dispose() {
    patchEmbed.dispose();
    posEmbed.dispose();
    for (final b in blocks) {
      b.dispose();
    }
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// MLP Projector (vision dim → language model dim)
// ---------------------------------------------------------------------------

final class _MlpProjector extends Module {
  _MlpProjector(MLXContext ctx, Lfm2Config cfg)
      : linear1 = Linear(
          ctx,
          inFeatures: cfg.visionConfig.hiddenDim,
          outFeatures: cfg.effectiveProjectorHiddenDim,
          bias: true,
        ),
        linear2 = Linear(
          ctx,
          inFeatures: cfg.effectiveProjectorHiddenDim,
          outFeatures: cfg.hiddenDim,
          bias: true,
        );

  Linear linear1;
  Linear linear2;

  MLXArray call(MLXArray x) {
    final h = linear1.call(x);
    final act = h.gelu();
    h.dispose();
    final out = linear2.call(act);
    act.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...linear1.parameters().map((k, v) => MapEntry('linear1.$k', v)),
        ...linear2.parameters().map((k, v) => MapEntry('linear2.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    linear1.loadWeights(_scoped(weights, 'linear1'));
    linear2.loadWeights(_scoped(weights, 'linear2'));
  }

  @override
  void dispose() {
    linear1.dispose();
    linear2.dispose();
  }
}

// ---------------------------------------------------------------------------
// Mamba SSM Block (Mamba-1 formulation)
// ---------------------------------------------------------------------------
//
// Architecture (per https://arxiv.org/abs/2312.00752):
//   in_proj: d_model → 2 * d_inner    (z and x)
//   depthwise conv1d: d_inner, kernel k
//   x1_proj: d_inner → dt_rank + 2*d_state   (dt, B, C)
//   dt_proj: dt_rank → d_inner        (with bias, initialised to dt-softplus)
//   selective_scan over d_inner states of size d_state
//   out_proj: d_inner → d_model
//

final class _MambaBlock extends Module {
  _MambaBlock(MLXContext ctx, Lfm2Config cfg)
      : inProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: 2 * cfg.ssmInnerDim,
          bias: false,
        ),
        convWeight = MLXArray.zeros(ctx, [cfg.ssmInnerDim, cfg.ssmConvSize]),
        convBias = MLXArray.zeros(ctx, [cfg.ssmInnerDim]),
        x1Proj = Linear(
          ctx,
          inFeatures: cfg.ssmInnerDim,
          outFeatures: cfg.ssmDtRank + 2 * cfg.ssmDState,
          bias: false,
        ),
        dtProj = Linear(
          ctx,
          inFeatures: cfg.ssmDtRank,
          outFeatures: cfg.ssmInnerDim,
          bias: true,
        ),
        outProj = Linear(
          ctx,
          inFeatures: cfg.ssmInnerDim,
          outFeatures: cfg.hiddenDim,
          bias: false,
        ),
        norm = RMSNorm(ctx, dims: cfg.hiddenDim, eps: cfg.rmsNormEps),
        aLog = MLXArray.zeros(ctx, [cfg.ssmInnerDim, cfg.ssmDState]),
        d = MLXArray.ones(ctx, [cfg.ssmInnerDim]),
        _dState = cfg.ssmDState,
        _convSize = cfg.ssmConvSize;

  Linear inProj;
  // Depthwise conv1d stored as [d_inner, conv_size] weight + [d_inner] bias.
  MLXArray convWeight;
  MLXArray convBias;
  Linear x1Proj;
  Linear dtProj;
  Linear outProj;
  RMSNorm norm;
  // A_log shape [d_inner, d_state]; actual A = -exp(A_log) (negative for stability).
  MLXArray aLog;
  // Skip-connection coefficient D, shape [d_inner].
  MLXArray d;

  final int _dState;
  final int _convSize;

  MLXArray call(MLXContext ctx, MLXArray x, SsmCache cache) {
    // x: [B, L, d_model]
    final residual = x;
    final normed = norm.call(x);

    final projected = inProj.call(normed);
    normed.dispose();

    // Split into gate z and ssm input xIn  [B, L, d_inner] each.
    final dInner = projected.dim(-1) ~/ 2;
    final parts = projected.split(2, axis: -1);
    projected.dispose();
    final z = parts[0]; // gate
    final xIn = parts[1];

    // Depthwise conv1d with causal padding + SiLU.
    final xConv = _conv1d(ctx, xIn, cache);
    xIn.dispose();
    final xSilu = xConv.silu();
    xConv.dispose();

    // Project to dt, B, C.
    final x1Out = x1Proj.call(xSilu);
    final x1Parts = x1Out.split(3, axis: -1);
    x1Out.dispose();
    final dtRaw = x1Parts[0];  // [B, L, dt_rank]
    final bProj = x1Parts[1];  // [B, L, d_state]
    final cProj = x1Parts[2];  // [B, L, d_state]

    // dt: softplus after projecting rank → d_inner.
    final dtFull = dtProj.call(dtRaw);
    dtRaw.dispose();
    final dt = _softplus(ctx, dtFull);
    dtFull.dispose();

    // A = -exp(A_log): [d_inner, d_state].
    final negALog = MLXArray.zeros(ctx, [1]) - aLog;
    final aNeg = negALog.exp();
    negALog.dispose();

    // Selective scan.
    final y = _selectiveScan(ctx, xSilu, dt, aNeg, bProj, cProj, cache);
    xSilu.dispose();
    dt.dispose();
    aNeg.dispose();
    bProj.dispose();
    cProj.dispose();

    // Add skip connection D * xSilu (already disposed; use y directly).
    // y shape [B, L, d_inner]

    // Gate with z via silu.
    final zSilu = z.silu();
    z.dispose();
    final gated = y * zSilu;
    y.dispose();
    zSilu.dispose();
    assert(dInner > 0);

    final out = outProj.call(gated);
    gated.dispose();

    // Residual.
    final result = residual + out;
    out.dispose();
    return result;
  }

  // ---------------------------------------------------------------------------
  // Depthwise conv1d with SsmCache state buffer.
  //
  // convWeight: [d_inner, conv_size]
  // convBias:   [d_inner]
  // cache.convState: [1, d_inner, conv_size] (null on first call)
  //
  // For multi-token input (prefill), processes tokens sequentially and updates
  // the state buffer so that generation can continue correctly.
  // ---------------------------------------------------------------------------
  MLXArray _conv1d(MLXContext ctx, MLXArray xIn, SsmCache cache) {
    final b = xIn.dim(0);
    final l = xIn.dim(1);
    final dInner = xIn.dim(2);

    // Initialise conv state if needed: [B, d_inner, conv_size] of zeros.
    cache.convState ??= MLXArray.zeros(ctx, [b, dInner, _convSize]);

    final outputs = <MLXArray>[];

    for (var t = 0; t < l; t++) {
      // token: [B, d_inner]
      final token = xIn.slice(
        start: [0, t, 0],
        stop: [b, t + 1, dInner],
        strides: [1, 1, 1],
      ).squeeze(axis: 1);

      // Shift conv state left and append new token.
      final oldState = cache.convState!;
      final tail = oldState.slice(
        start: [0, 0, 1],
        stop: [b, dInner, _convSize],
        strides: [1, 1, 1],
      ); // [B, d_inner, conv_size-1]
      final tokenExp = token.expandDims(-1); // [B, d_inner, 1]
      final newState = concatenate(ctx, [tail, tokenExp], axis: -1);
      tail.dispose();
      tokenExp.dispose();
      token.dispose();
      oldState.dispose();
      cache.convState = newState;

      // Apply depthwise conv: sum over kernel dimension.
      // newState: [B, d_inner, conv_size], convWeight: [d_inner, conv_size]
      final wExp = convWeight.expandDims(0); // [1, d_inner, conv_size]
      final prod = newState * wExp;
      wExp.dispose();
      final summed = prod.sum(axis: -1); // [B, d_inner]
      prod.dispose();
      final biased = summed + convBias;
      summed.dispose();
      outputs.add(biased);
    }

    // Stack outputs: [B, L, d_inner]
    final stacked = stack(ctx, outputs, axis: 1);
    for (final o in outputs) {
      o.dispose();
    }
    return stacked;
  }

  // ---------------------------------------------------------------------------
  // Selective scan (sequential recurrence).
  //
  // x:  [B, L, d_inner]
  // dt: [B, L, d_inner]
  // a:  [d_inner, d_state]  (already negated: a = exp(-A_log))
  // B:  [B, L, d_state]
  // C:  [B, L, d_state]
  //
  // ssm_state: [B, d_inner, d_state]
  // ---------------------------------------------------------------------------
  MLXArray _selectiveScan(
    MLXContext ctx,
    MLXArray x,
    MLXArray dt,
    MLXArray a,
    MLXArray bProj,
    MLXArray cProj,
    SsmCache cache,
  ) {
    final b = x.dim(0);
    final l = x.dim(1);
    final dInner = x.dim(2);

    cache.ssmState ??= MLXArray.zeros(ctx, [b, dInner, _dState]);

    final outputs = <MLXArray>[];

    for (var t = 0; t < l; t++) {
      // xt: [B, d_inner], dtt: [B, d_inner]
      final xt = x
          .slice(start: [0, t, 0], stop: [b, t + 1, dInner], strides: [1, 1, 1])
          .squeeze(axis: 1);
      final dtt = dt
          .slice(start: [0, t, 0], stop: [b, t + 1, dInner], strides: [1, 1, 1])
          .squeeze(axis: 1);
      // bt, ct: [B, d_state]
      final bt = bProj
          .slice(start: [0, t, 0], stop: [b, t + 1, _dState], strides: [1, 1, 1])
          .squeeze(axis: 1);
      final ct = cProj
          .slice(start: [0, t, 0], stop: [b, t + 1, _dState], strides: [1, 1, 1])
          .squeeze(axis: 1);

      // discrete_A = exp(dt * A)  [B, d_inner, d_state]
      // dt: [B, d_inner] → [B, d_inner, 1]
      // a:  [d_inner, d_state] → [1, d_inner, d_state]
      final dtExp = dtt.expandDims(-1); // [B, d_inner, 1]
      final aExp = a.expandDims(0); // [1, d_inner, d_state]
      final dtA = dtExp * aExp;
      dtExp.dispose();
      aExp.dispose();
      final discA = dtA.exp();
      dtA.dispose();

      // discrete_B_t = dt_t * B_t  → outer product [B, d_inner, d_state]
      // dtt: [B, d_inner] → [B, d_inner, 1]
      // bt:  [B, d_state] → [B, 1, d_state]
      final dtExp2 = dtt.expandDims(-1);
      final btExp = bt.expandDims(1);
      final discB = dtExp2 * btExp;
      dtExp2.dispose();
      btExp.dispose();

      // xt: [B, d_inner] → [B, d_inner, 1]
      final xtExp = xt.expandDims(-1);

      // new_ssm_state = discA * old + discB * x_t
      final oldState = cache.ssmState!;
      final term1 = discA * oldState;
      discA.dispose();
      final term2 = discB * xtExp;
      discB.dispose();
      xtExp.dispose();
      final newState = term1 + term2;
      term1.dispose();
      term2.dispose();
      oldState.dispose();
      cache.ssmState = newState;

      // y_t = einsum('bds,bs->bd', newState, ct) + D * xt
      // newState: [B, d_inner, d_state], ct: [B, d_state]
      // → matmul: [B, d_inner, d_state] x [B, d_state, 1] = [B, d_inner, 1]
      final ctExp = ct.expandDims(-1); // [B, d_state, 1]
      final yMm = newState.matmul(ctExp).squeeze(axis: -1); // [B, d_inner]
      ctExp.dispose();

      // skip connection: D * xt
      final dSkip = d * xt;
      final yt = yMm + dSkip;
      yMm.dispose();
      dSkip.dispose();

      xt.dispose();
      dtt.dispose();
      bt.dispose();
      ct.dispose();
      outputs.add(yt);
    }

    final stacked = stack(ctx, outputs, axis: 1); // [B, L, d_inner]
    for (final o in outputs) {
      o.dispose();
    }
    return stacked;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...inProj.parameters().map((k, v) => MapEntry('in_proj.$k', v)),
        'conv_weight': convWeight,
        'conv_bias': convBias,
        ...x1Proj.parameters().map((k, v) => MapEntry('x1_proj.$k', v)),
        ...dtProj.parameters().map((k, v) => MapEntry('dt_proj.$k', v)),
        ...outProj.parameters().map((k, v) => MapEntry('out_proj.$k', v)),
        ...norm.parameters().map((k, v) => MapEntry('norm.$k', v)),
        'A_log': aLog,
        'D': d,
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    inProj.loadWeights(_scoped(weights, 'in_proj'));
    if (weights['conv_weight'] case final w?) convWeight = w;
    if (weights['conv_bias'] case final b?) convBias = b;
    x1Proj.loadWeights(_scoped(weights, 'x1_proj'));
    dtProj.loadWeights(_scoped(weights, 'dt_proj'));
    outProj.loadWeights(_scoped(weights, 'out_proj'));
    norm.loadWeights(_scoped(weights, 'norm'));
    if (weights['A_log'] case final a?) aLog = a;
    if (weights['D'] case final dd?) d = dd;
  }

  @override
  void dispose() {
    inProj.dispose();
    convWeight.dispose();
    convBias.dispose();
    x1Proj.dispose();
    dtProj.dispose();
    outProj.dispose();
    norm.dispose();
    aLog.dispose();
    d.dispose();
  }
}

// ---------------------------------------------------------------------------
// Attention Block (GQA + RoPE)
// ---------------------------------------------------------------------------

final class _Lfm2Attention extends Module {
  _Lfm2Attention(MLXContext ctx, Lfm2Config cfg)
      : qProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.numAttentionHeads * cfg.headDim,
          bias: false,
        ),
        kProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.numKVHeads * cfg.headDim,
          bias: false,
        ),
        vProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.numKVHeads * cfg.headDim,
          bias: false,
        ),
        oProj = Linear(
          ctx,
          inFeatures: cfg.numAttentionHeads * cfg.headDim,
          outFeatures: cfg.hiddenDim,
          bias: false,
        ),
        norm = RMSNorm(ctx, dims: cfg.hiddenDim, eps: cfg.rmsNormEps),
        numHeads = cfg.numAttentionHeads,
        numKVHeads = cfg.numKVHeads,
        headDim = cfg.headDim,
        scale = 1.0 / math.sqrt(cfg.headDim.toDouble()),
        ropeTheta = cfg.ropeTheta;

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  RMSNorm norm;
  final int numHeads;
  final int numKVHeads;
  final int headDim;
  final double scale;
  final double ropeTheta;

  MLXArray call(
    MLXContext ctx,
    MLXArray x,
    KVCache cache, {
    MLXArray? mask,
  }) {
    final residual = x;
    final normed = norm.call(x);

    final b = normed.dim(0);
    final l = normed.dim(1);

    var q = qProj
        .call(normed)
        .reshape([b, l, numHeads, headDim])
        .transpose([0, 2, 1, 3]);
    var k = kProj
        .call(normed)
        .reshape([b, l, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    final v = vProj
        .call(normed)
        .reshape([b, l, numKVHeads, headDim])
        .transpose([0, 2, 1, 3]);
    normed.dispose();

    // Apply RoPE.
    q = q.rope(dims: headDim, base: ropeTheta, offset: cache.offset);
    k = k.rope(dims: headDim, base: ropeTheta, offset: cache.offset);

    // KV cache update.
    final (keys, values) = cache.update(k, v);

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: q,
      keys: keys,
      values: values,
      scale: scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    q.dispose();

    final merged =
        attnOut.transpose([0, 2, 1, 3]).reshape([b, l, numHeads * headDim]);
    attnOut.dispose();
    final out = oProj.call(merged);
    merged.dispose();

    final result = residual + out;
    out.dispose();
    return result;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...qProj.parameters().map((k, v) => MapEntry('q_proj.$k', v)),
        ...kProj.parameters().map((k, v) => MapEntry('k_proj.$k', v)),
        ...vProj.parameters().map((k, v) => MapEntry('v_proj.$k', v)),
        ...oProj.parameters().map((k, v) => MapEntry('o_proj.$k', v)),
        ...norm.parameters().map((k, v) => MapEntry('input_layernorm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    oProj.loadWeights(_scoped(weights, 'o_proj'));
    norm.loadWeights(_scoped(weights, 'input_layernorm'));
  }

  @override
  void dispose() {
    qProj.dispose();
    kProj.dispose();
    vProj.dispose();
    oProj.dispose();
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP (used only with attention layers in some variants)
// ---------------------------------------------------------------------------

final class _Lfm2MLP extends Module {
  _Lfm2MLP(MLXContext ctx, Lfm2Config cfg)
      : gateProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.mlpDim,
          bias: false,
        ),
        downProj = Linear(
          ctx,
          inFeatures: cfg.mlpDim,
          outFeatures: cfg.hiddenDim,
          bias: false,
        ),
        upProj = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.mlpDim,
          bias: false,
        ),
        norm = RMSNorm(ctx, dims: cfg.hiddenDim, eps: cfg.rmsNormEps);

  Linear gateProj;
  Linear downProj;
  Linear upProj;
  RMSNorm norm;

  MLXArray call(MLXArray x) {
    final residual = x;
    final normed = norm.call(x);
    final gate = gateProj.call(normed).silu();
    final up = upProj.call(normed);
    normed.dispose();
    final hidden = gate * up;
    gate.dispose();
    up.dispose();
    final out = downProj.call(hidden);
    hidden.dispose();
    final result = residual + out;
    out.dispose();
    return result;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...gateProj.parameters().map((k, v) => MapEntry('gate_proj.$k', v)),
        ...downProj.parameters().map((k, v) => MapEntry('down_proj.$k', v)),
        ...upProj.parameters().map((k, v) => MapEntry('up_proj.$k', v)),
        ...norm.parameters().map((k, v) => MapEntry('post_attention_layernorm.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    gateProj.loadWeights(_scoped(weights, 'gate_proj'));
    downProj.loadWeights(_scoped(weights, 'down_proj'));
    upProj.loadWeights(_scoped(weights, 'up_proj'));
    norm.loadWeights(_scoped(weights, 'post_attention_layernorm'));
  }

  @override
  void dispose() {
    gateProj.dispose();
    downProj.dispose();
    upProj.dispose();
    norm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Decoder layer — either Mamba SSM or Attention + MLP
// ---------------------------------------------------------------------------

final class _Lfm2DecoderLayer extends Module {
  _Lfm2DecoderLayer.ssm(MLXContext ctx, Lfm2Config cfg)
      : mamba = _MambaBlock(ctx, cfg),
        attn = null,
        mlp = null;

  _Lfm2DecoderLayer.attention(MLXContext ctx, Lfm2Config cfg)
      : attn = _Lfm2Attention(ctx, cfg),
        mlp = _Lfm2MLP(ctx, cfg),
        mamba = null;

  final _MambaBlock? mamba;
  final _Lfm2Attention? attn;
  final _Lfm2MLP? mlp;

  bool get isAttn => attn != null;

  MLXArray call(
    MLXContext ctx,
    MLXArray x,
    KVCache cache, {
    MLXArray? mask,
  }) {
    if (mamba case final m?) {
      return m.call(ctx, x, cache as SsmCache);
    }
    // Attention + MLP.
    final afterAttn = attn!.call(ctx, x, cache, mask: mask);
    final afterMlp = mlp!.call(afterAttn);
    afterAttn.dispose();
    return afterMlp;
  }

  @override
  Map<String, MLXArray> parameters() {
    if (mamba case final m?) {
      return m.parameters();
    }
    return {
      ...attn!.parameters(),
      ...mlp!.parameters(),
    };
  }

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    if (mamba case final m?) {
      m.loadWeights(weights);
    } else {
      attn!.loadWeights(weights);
      mlp!.loadWeights(weights);
    }
  }

  @override
  void dispose() {
    mamba?.dispose();
    attn?.dispose();
    mlp?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Full LFM2.4 Vision-Language Model
// ---------------------------------------------------------------------------

final class Lfm2VLModel extends Module
    implements VLMModel, KVCacheDimensionProvider {
  Lfm2VLModel(MLXContext ctx, Lfm2Config cfg)
      : _cfg = cfg,
        embed = Embedding(ctx, vocabSize: cfg.vocabSize, dims: cfg.hiddenDim),
        _visionEncoder = _SigLIPEncoder(ctx, cfg.visionConfig),
        _projector = _MlpProjector(ctx, cfg),
        _layers = List.generate(cfg.numLayers, (i) {
          if (cfg.attentionLayerIndices.contains(i)) {
            return _Lfm2DecoderLayer.attention(ctx, cfg);
          }
          return _Lfm2DecoderLayer.ssm(ctx, cfg);
        }),
        norm = RMSNorm(ctx, dims: cfg.hiddenDim, eps: cfg.rmsNormEps),
        lmHead = Linear(
          ctx,
          inFeatures: cfg.hiddenDim,
          outFeatures: cfg.vocabSize,
          bias: false,
        ),
        _ctx = ctx;

  final Lfm2Config _cfg;
  final MLXContext _ctx;

  Embedding embed;
  final _SigLIPEncoder _visionEncoder;
  final _MlpProjector _projector;
  final List<_Lfm2DecoderLayer> _layers;
  RMSNorm norm;
  Linear lmHead;

  // ---------------------------------------------------------------------------
  // VLMModel interface
  // ---------------------------------------------------------------------------

  @override
  VisionModel get visionModel => _visionEncoder;

  @override
  LMOutput callWithPixels(
    MLXContext ctx,
    LMInput input, {
    List<KVCache>? cache,
    LMState? state,
  }) {
    final textTokens = input.text.tokens;
    final b = textTokens.dim(0);
    final l = textTokens.dim(1);

    // Embed text tokens.
    var hiddenStates = embed.call(textTokens);

    // Merge vision features if an image is present.
    if (input.image case final img?) {
      final visualFeatures = _visionEncoder.encodeImage(ctx, img.pixels);
      final projected = _projector.call(visualFeatures);
      visualFeatures.dispose();

      // Vision tokens replace the first numVisionTokens positions.
      // In practice the caller should have set up token IDs so that the
      // image placeholder positions correspond to the visual features.
      // Here we prepend them before the text embeddings (simple strategy).
      final combined = concatenate(ctx, [projected, hiddenStates], axis: 1);
      projected.dispose();
      hiddenStates.dispose();
      hiddenStates = combined;
    }

    final mask = _makeCausalMask(ctx, cache: cache, b: b, l: l);

    for (var i = 0; i < _layers.length; i++) {
      final next = _layers[i].call(ctx, hiddenStates, cache![i], mask: mask);
      hiddenStates.dispose();
      hiddenStates = next;
    }
    mask?.dispose();

    final normed = norm.call(hiddenStates);
    hiddenStates.dispose();
    final logits = lmHead.call(normed);
    normed.dispose();

    return LMOutput(logits: logits);
  }

  // ---------------------------------------------------------------------------
  // LanguageModel interface
  // ---------------------------------------------------------------------------

  @override
  PrepareResult prepare(LMInput input, List<KVCache> cache, {int? windowSize}) {
    if (input.image != null) {
      // Run full forward with image merged.
      final out = callWithPixels(_ctx, input, cache: cache);
      return PrepareResult.logits(out);
    }
    return PrepareResult.tokens(input.text);
  }

  @override
  LMOutput call(LMInputText input, {List<KVCache>? cache, LMState? state}) {
    final b = input.tokens.dim(0);
    final l = input.tokens.dim(1);

    var x = embed.call(input.tokens);
    final mask = _makeCausalMask(_ctx, cache: cache, b: b, l: l);

    for (var i = 0; i < _layers.length; i++) {
      final next = _layers[i].call(_ctx, x, cache![i], mask: mask);
      x.dispose();
      x = next;
    }
    mask?.dispose();

    final normed = norm.call(x);
    x.dispose();
    final logits = lmHead.call(normed);
    normed.dispose();

    return LMOutput(logits: logits);
  }

  @override
  List<KVCache> newCache(GenerateParameters? parameters) {
    return List.generate(_cfg.numLayers, (i) {
      if (_cfg.attentionLayerIndices.contains(i)) {
        if (parameters?.maxKVSize case final maxKV?) {
          return RotatingKVCache(maxSize: maxKV);
        }
        return KVCacheSimple();
      }
      return SsmCache();
    });
  }

  @override
  List<int> get kvHeads => List.generate(
        _cfg.numLayers,
        (i) => _cfg.attentionLayerIndices.contains(i) ? _cfg.numKVHeads : 0,
      );

  @override
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) {
    final out = <String, MLXArray>{};
    for (final e in weights.entries) {
      // Skip tied lm_head if it's the same as embed.
      if (e.key == 'lm_head.weight' && weights.containsKey('model.embed_tokens.weight')) {
        final embedW = weights['model.embed_tokens.weight']!;
        if (e.value.shape.toString() == embedW.shape.toString()) continue;
      }
      out[e.key] = e.value;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Module interface
  // ---------------------------------------------------------------------------

  @override
  Map<String, MLXArray> parameters() => {
        ...embed.parameters().map((k, v) => MapEntry('model.embed_tokens.$k', v)),
        for (var i = 0; i < _layers.length; i++)
          ..._layers[i].parameters().map((k, v) => MapEntry('model.layers.$i.$k', v)),
        ...norm.parameters().map((k, v) => MapEntry('model.norm.$k', v)),
        ...lmHead.parameters().map((k, v) => MapEntry('lm_head.$k', v)),
        ..._visionEncoder.parameters().map((k, v) => MapEntry('vision_model.$k', v)),
        ..._projector.parameters().map((k, v) => MapEntry('multi_modal_projector.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    embed.loadWeights(_scoped(weights, 'model.embed_tokens'));
    for (var i = 0; i < _layers.length; i++) {
      _layers[i].loadWeights(_scoped(weights, 'model.layers.$i'));
    }
    norm.loadWeights(_scoped(weights, 'model.norm'));
    lmHead.loadWeights(_scoped(weights, 'lm_head'));
    _visionEncoder.loadWeights(_scoped(weights, 'vision_model'));
    _projector.loadWeights(_scoped(weights, 'multi_modal_projector'));
  }

  @override
  void dispose() {
    embed.dispose();
    _visionEncoder.dispose();
    _projector.dispose();
    for (final l in _layers) {
      l.dispose();
    }
    norm.dispose();
    lmHead.dispose();
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  MLXArray? _makeCausalMask(
    MLXContext ctx, {
    required List<KVCache>? cache,
    required int b,
    required int l,
  }) {
    if (l == 1) return null;
    // Use the first attention-layer cache for the offset.
    final attnIdx = _cfg.attentionLayerIndices.isEmpty
        ? 0
        : _cfg.attentionLayerIndices.first;
    final offset = cache?[attnIdx].offset ?? 0;
    return createCausalMask(_ctx, n: l, offset: offset);
    // Suppress unused variable warnings.
  }
}
