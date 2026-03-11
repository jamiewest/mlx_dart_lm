import 'dart:math' as math;

import 'package:mlx_dart/mlx_dart.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Map<String, MLXArray> _scoped(Map<String, MLXArray> w, String prefix) {
  final p = '$prefix.';
  return {
    for (final e in w.entries)
      if (e.key.startsWith(p)) e.key.substring(p.length): e.value,
  };
}

// ---------------------------------------------------------------------------
// VitMLP — two-layer MLP used inside vision transformer blocks
// ---------------------------------------------------------------------------

final class VitMLP extends Module {
  VitMLP(
    MLXContext ctx, {
    required int dims,
    required int hiddenDims,
    bool useGelu = true,
  })  : fc1 = Linear(ctx, inFeatures: dims, outFeatures: hiddenDims),
        fc2 = Linear(ctx, inFeatures: hiddenDims, outFeatures: dims),
        _useGelu = useGelu;

  Linear fc1;
  Linear fc2;
  final bool _useGelu;

  MLXArray call(MLXArray x) {
    final h = fc1.call(x);
    final act = _useGelu ? h.gelu() : h.silu();
    h.dispose();
    final out = fc2.call(act);
    act.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...fc1.parameters().map((k, v) => MapEntry('fc1.$k', v)),
        ...fc2.parameters().map((k, v) => MapEntry('fc2.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    fc1.loadWeights(_scoped(weights, 'fc1'));
    fc2.loadWeights(_scoped(weights, 'fc2'));
  }

  @override
  void dispose() {
    fc1.dispose();
    fc2.dispose();
  }
}

// ---------------------------------------------------------------------------
// VitAttention — multi-head self-attention for vision transformers
// ---------------------------------------------------------------------------

final class VitAttention extends Module {
  VitAttention(
    MLXContext ctx, {
    required int dims,
    required this.numHeads,
    bool bias = true,
  })  : qProj = Linear(ctx, inFeatures: dims, outFeatures: dims, bias: bias),
        kProj = Linear(ctx, inFeatures: dims, outFeatures: dims, bias: bias),
        vProj = Linear(ctx, inFeatures: dims, outFeatures: dims, bias: bias),
        oProj = Linear(ctx, inFeatures: dims, outFeatures: dims, bias: bias),
        headDim = dims ~/ numHeads,
        scale = 1.0 / math.sqrt((dims ~/ numHeads).toDouble());

  Linear qProj;
  Linear kProj;
  Linear vProj;
  Linear oProj;
  final int numHeads;
  final int headDim;
  final double scale;

  MLXArray call(MLXContext ctx, MLXArray x, {MLXArray? mask}) {
    final b = x.dim(0);
    final s = x.dim(1);

    final q = qProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final k = kProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);
    final v = vProj.call(x).reshape([b, s, numHeads, headDim]).transpose([0, 2, 1, 3]);

    final attnOut = scaledDotProductAttention(
      ctx,
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      maskMode: mask != null ? 'array' : 'none',
      mask: mask,
    );
    q.dispose();

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
        ...oProj.parameters().map((k, v) => MapEntry('out_proj.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    qProj.loadWeights(_scoped(weights, 'q_proj'));
    kProj.loadWeights(_scoped(weights, 'k_proj'));
    vProj.loadWeights(_scoped(weights, 'v_proj'));
    // Some checkpoints use 'out_proj', others 'o_proj'.
    final oScope = weights.containsKey('out_proj.weight') ? 'out_proj' : 'o_proj';
    oProj.loadWeights(_scoped(weights, oScope));
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
// VitEncoderBlock — single ViT layer (pre-norm, attn + MLP)
// ---------------------------------------------------------------------------

final class VitEncoderBlock extends Module {
  VitEncoderBlock(
    MLXContext ctx, {
    required int dims,
    required int numHeads,
    required int mlpDims,
    double eps = 1e-6,
    bool attentionBias = true,
    bool useGelu = true,
  })  : norm1 = LayerNorm(ctx, dims: dims, eps: eps, bias: true),
        norm2 = LayerNorm(ctx, dims: dims, eps: eps, bias: true),
        attn = VitAttention(ctx, dims: dims, numHeads: numHeads, bias: attentionBias),
        mlp = VitMLP(ctx, dims: dims, hiddenDims: mlpDims, useGelu: useGelu);

  LayerNorm norm1;
  LayerNorm norm2;
  VitAttention attn;
  VitMLP mlp;

  MLXArray call(MLXContext ctx, MLXArray x, {MLXArray? mask}) {
    final normed1 = norm1.call(x);
    final attnOut = attn.call(ctx, normed1, mask: mask);
    normed1.dispose();
    final afterAttn = x + attnOut;
    attnOut.dispose();

    final normed2 = norm2.call(afterAttn);
    final mlpOut = mlp.call(normed2);
    normed2.dispose();
    final out = afterAttn + mlpOut;
    mlpOut.dispose();
    afterAttn.dispose();
    return out;
  }

  @override
  Map<String, MLXArray> parameters() => {
        ...norm1.parameters().map((k, v) => MapEntry('layer_norm1.$k', v)),
        ...norm2.parameters().map((k, v) => MapEntry('layer_norm2.$k', v)),
        ...attn.parameters().map((k, v) => MapEntry('self_attn.$k', v)),
        ...mlp.parameters().map((k, v) => MapEntry('mlp.$k', v)),
      };

  @override
  void loadWeights(Map<String, MLXArray> weights) {
    norm1.loadWeights(_scoped(weights, 'layer_norm1'));
    norm2.loadWeights(_scoped(weights, 'layer_norm2'));
    attn.loadWeights(_scoped(weights, 'self_attn'));
    mlp.loadWeights(_scoped(weights, 'mlp'));
  }

  @override
  void dispose() {
    norm1.dispose();
    norm2.dispose();
    attn.dispose();
    mlp.dispose();
  }
}
