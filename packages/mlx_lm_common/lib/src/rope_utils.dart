import 'dart:math' as math;

import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// RopeLayer interface
// ---------------------------------------------------------------------------

/// A callable that applies Rotary Position Embedding to an input tensor.
///
/// Mirrors the `RoPELayer` typealias from mlx-swift-lm.
abstract interface class RopeLayer {
  /// Apply RoPE to [x] (shape `[B, H, S, headDim]`) with token [offset].
  MLXArray call(MLXArray x, int offset);

  void dispose();
}

// ---------------------------------------------------------------------------
// DefaultRope
// ---------------------------------------------------------------------------

/// Standard RoPE with fixed base and optional linear scaling.
///
/// Handles `rope_type: "default"` and `rope_type: "linear"`.
final class DefaultRope implements RopeLayer {
  const DefaultRope({
    required this.dims,
    this.traditional = false,
    required this.base,
    this.scale = 1.0,
  });

  final int dims;
  final bool traditional;
  final double base;
  final double scale;

  @override
  MLXArray call(MLXArray x, int offset) =>
      x.rope(dims: dims, traditional: traditional, base: base, scale: scale, offset: offset);

  @override
  void dispose() {}
}

// ---------------------------------------------------------------------------
// Llama3Rope
// ---------------------------------------------------------------------------

/// Llama 3 RoPE with frequency-domain scaling.
///
/// Applies smooth rescaling to frequencies based on wavelength thresholds so
/// that long-range tokens can attend across the extended context window.
///
/// Mirrors `Llama3RoPE` from mlx-swift-lm.
final class Llama3Rope implements RopeLayer {
  Llama3Rope(
    MLXContext ctx, {
    required this.dims,
    this.traditional = false,
    double base = 500000.0,
    required Map<String, dynamic> scalingConfig,
  }) : _freqs = _computeFreqs(ctx, dims, base, scalingConfig);

  final int dims;
  final bool traditional;
  final MLXArray _freqs;

  static MLXArray _computeFreqs(
    MLXContext ctx,
    int dims,
    double base,
    Map<String, dynamic> scalingConfig,
  ) {
    final factor = (scalingConfig['factor'] as num?)?.toDouble() ?? 1.0;
    final lowFreqFactor = (scalingConfig['low_freq_factor'] as num?)?.toDouble() ?? 1.0;
    final highFreqFactor = (scalingConfig['high_freq_factor'] as num?)?.toDouble() ?? 4.0;
    final oldContextLen =
        (scalingConfig['original_max_position_embeddings'] as num?)?.toDouble() ?? 8192.0;

    final lowFreqWavelen = oldContextLen / lowFreqFactor;
    final highFreqWavelen = oldContextLen / highFreqFactor;

    // frequencies = base^(i/dims) for i in [0, 2, 4, ..., dims-2]
    final indices = MLXArray.arange(ctx, 0, dims.toDouble(), 2.0, dtype: MLXDtype.float32);
    final logBase = MLXArray.float_(ctx, math.log(base));
    final dimsArr = MLXArray.float_(ctx, dims.toDouble());
    final t1 = indices / dimsArr;
    final t2 = t1 * logBase;
    var frequencies = t2.exp();
    indices.dispose();
    dimsArr.dispose();
    logBase.dispose();
    t1.dispose();
    t2.dispose();

    // wavelens = 2π * frequencies
    final twoPi = MLXArray.float_(ctx, 2.0 * math.pi);
    final wavelens = twoPi * frequencies;
    twoPi.dispose();

    // Low-frequency: wavelen > lowFreqWavelen → multiply by factor
    final lowWavelArr = MLXArray.float_(ctx, lowFreqWavelen);
    final lowMask = wavelens.greater(lowWavelArr);
    lowWavelArr.dispose();
    final factorArr = MLXArray.float_(ctx, factor);
    final scaledLow = frequencies * factorArr;
    final freqs1 = where(ctx, lowMask, scaledLow, frequencies);
    scaledLow.dispose();
    lowMask.dispose();
    frequencies.dispose();
    frequencies = freqs1;

    // Medium-frequency: highFreqWavelen < wavelen < lowFreqWavelen → smooth blend
    final highWavelArr = MLXArray.float_(ctx, highFreqWavelen);
    final lowWavelArr2 = MLXArray.float_(ctx, lowFreqWavelen);
    final isMedium = wavelens.greater(highWavelArr).logicalAnd(wavelens.less(lowWavelArr2));
    highWavelArr.dispose();
    lowWavelArr2.dispose();

    // smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
    final oldCtxArr = MLXArray.float_(ctx, oldContextLen);
    final lfArr = MLXArray.float_(ctx, lowFreqFactor);
    final rangeArr = MLXArray.float_(ctx, highFreqFactor - lowFreqFactor);
    final smoothFactors = (oldCtxArr / wavelens - lfArr) / rangeArr;
    oldCtxArr.dispose();
    lfArr.dispose();
    rangeArr.dispose();
    wavelens.dispose();

    // smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)
    final one = MLXArray.float_(ctx, 1.0);
    final oneMinusSmooth = one - smoothFactors;
    one.dispose();
    final blendDenom = oneMinusSmooth / factorArr + smoothFactors;
    factorArr.dispose();
    oneMinusSmooth.dispose();
    smoothFactors.dispose();
    final smoothFreqs = frequencies / blendDenom;
    blendDenom.dispose();

    final result = where(ctx, isMedium, smoothFreqs, frequencies);
    isMedium.dispose();
    smoothFreqs.dispose();
    frequencies.dispose();

    return result;
  }

  @override
  MLXArray call(MLXArray x, int offset) =>
      x.rope(dims: dims, traditional: traditional, offset: offset, freqs: _freqs);

  @override
  void dispose() => _freqs.dispose();
}

// ---------------------------------------------------------------------------
// YarnRope
// ---------------------------------------------------------------------------

/// YaRN (Yet another RoPE extensioN) with extended context scaling.
///
/// Mirrors `YarnRoPE` from mlx-swift-lm.
final class YarnRope implements RopeLayer {
  YarnRope(
    MLXContext ctx, {
    required this.dims,
    this.traditional = false,
    double base = 500000.0,
    double scalingFactor = 1.0,
    int originalMaxPositionEmbeddings = 4096,
    double betaFast = 32.0,
    double betaSlow = 1.0,
    double mscale = 1.0,
    double mscaleAllDim = 0.0,
  })  : _mscale = _yarnGetMscale(scalingFactor, mscale) /
            _yarnGetMscale(scalingFactor, mscaleAllDim),
        _freqs = _computeFreqs(
          ctx,
          dims,
          base,
          scalingFactor,
          originalMaxPositionEmbeddings,
          betaFast,
          betaSlow,
        );

  final int dims;
  final bool traditional;
  final double _mscale;
  final MLXArray _freqs;

  static double _yarnGetMscale(double scale, double mscale) {
    if (scale <= 1.0) return 1.0;
    return 0.1 * mscale * math.log(scale) + 1.0;
  }

  static double _yarnFindCorrectionDim(
    int dims,
    double base,
    int origMaxPos,
    double numRotations,
  ) =>
      dims * math.log(origMaxPos / (numRotations * 2 * math.pi)) / (2 * math.log(base));

  static MLXArray _linearRampMask(MLXContext ctx, double minVal, double maxVal, int dim) {
    final maxV = maxVal == minVal ? maxVal + 0.001 : maxVal;
    final indices = MLXArray.arange(ctx, 0, dim.toDouble(), 1.0, dtype: MLXDtype.float32);
    final minA = MLXArray.float_(ctx, minVal);
    final rangeA = MLXArray.float_(ctx, maxV - minVal);
    final linear = (indices - minA) / rangeA;
    indices.dispose();
    minA.dispose();
    rangeA.dispose();
    // clamp to [0, 1]
    final zero = MLXArray.float_(ctx, 0.0);
    final one = MLXArray.float_(ctx, 1.0);
    final clamped = linear.maximum(zero).minimum(one);
    linear.dispose();
    zero.dispose();
    one.dispose();
    return clamped;
  }

  static MLXArray _computeFreqs(
    MLXContext ctx,
    int dims,
    double base,
    double scalingFactor,
    int origMaxPos,
    double betaFast,
    double betaSlow,
  ) {
    final halfDim = dims ~/ 2;

    final low = _yarnFindCorrectionDim(dims, base, origMaxPos, betaFast)
        .clamp(0, dims - 1)
        .toInt();
    final high = _yarnFindCorrectionDim(dims, base, origMaxPos, betaSlow)
        .clamp(0, dims - 1)
        .ceil()
        .toInt();

    // freqExtra = base^(i/dims) for i in [0, 2, ..., dims-2]
    final indices = MLXArray.arange(ctx, 0, dims.toDouble(), 2.0, dtype: MLXDtype.float32);
    final logBase = MLXArray.float_(ctx, math.log(base));
    final dimsArr = MLXArray.float_(ctx, dims.toDouble());
    final t1 = indices / dimsArr;
    final t2 = t1 * logBase;
    final freqExtra = t2.exp();
    indices.dispose();
    dimsArr.dispose();
    logBase.dispose();
    t1.dispose();
    t2.dispose();

    // freqInter = scalingFactor * freqExtra
    final sfArr = MLXArray.float_(ctx, scalingFactor);
    final freqInter = freqExtra * sfArr;
    sfArr.dispose();

    // ramp = linearRampMask(low, high, halfDim)
    final ramp = _linearRampMask(ctx, low.toDouble(), high.toDouble(), halfDim);

    // result = (freqInter * freqExtra) / (freqInter * (1-ramp) + freqExtra * ramp)
    final one = MLXArray.float_(ctx, 1.0);
    final oneMinusRamp = one - ramp;
    one.dispose();

    final numerator = freqInter * freqExtra;
    final d1 = freqInter * oneMinusRamp;
    final d2 = freqExtra * ramp;
    final denominator = d1 + d2;
    final result = numerator / denominator;

    freqExtra.dispose();
    freqInter.dispose();
    ramp.dispose();
    oneMinusRamp.dispose();
    numerator.dispose();
    d1.dispose();
    d2.dispose();
    denominator.dispose();

    return result;
  }

  @override
  MLXArray call(MLXArray x, int offset) {
    if (_mscale != 1.0) {
      final scaleArr = MLXArray.float_(x.context, _mscale);
      final scaled = x * scaleArr;
      scaleArr.dispose();
      final result = scaled.rope(dims: dims, traditional: traditional, offset: offset, freqs: _freqs);
      scaled.dispose();
      return result;
    }
    return x.rope(dims: dims, traditional: traditional, offset: offset, freqs: _freqs);
  }

  @override
  void dispose() => _freqs.dispose();
}

// ---------------------------------------------------------------------------
// SuScaledRope (LongRoPE)
// ---------------------------------------------------------------------------

/// Su-scaled (LongRoPE) RoPE with per-frequency long-context scale factors.
///
/// Each base frequency is multiplied by a learned `longFactor` vector, and
/// the input is scaled by a constant derived from the context-length ratio.
///
/// Mirrors `SuScaledRoPE` from mlx-swift-lm.
final class SuScaledRope implements RopeLayer {
  SuScaledRope(
    MLXContext ctx, {
    required this.dims,
    this.traditional = false,
    double base = 10000.0,
    required List<double> longFactor,
    required int maxPositionEmbeddings,
    required int originalMaxPositionEmbeddings,
    double? longMScale,
  })  : _freqs = _computeFreqs(ctx, dims, base, longFactor),
        _scale = longMScale ??
            _computeScale(maxPositionEmbeddings, originalMaxPositionEmbeddings);

  final int dims;
  final bool traditional;
  final MLXArray _freqs;
  final double _scale;

  static MLXArray _computeFreqs(
    MLXContext ctx,
    int dims,
    double base,
    List<double> longFactor,
  ) {
    final indices =
        MLXArray.arange(ctx, 0, dims.toDouble(), 2.0, dtype: MLXDtype.float32);
    final logBase = MLXArray.float_(ctx, math.log(base));
    final dimsArr = MLXArray.float_(ctx, dims.toDouble());
    final t1 = indices / dimsArr;
    final t2 = t1 * logBase;
    final freqs = t2.exp();
    indices.dispose();
    dimsArr.dispose();
    logBase.dispose();
    t1.dispose();
    t2.dispose();

    final lfArr =
        MLXArray.fromFloats(ctx, longFactor, dtype: MLXDtype.float32);
    final result = lfArr * freqs;
    lfArr.dispose();
    freqs.dispose();
    return result;
  }

  static double _computeScale(
    int maxPositionEmbeddings,
    int originalMaxPositionEmbeddings,
  ) {
    final factor = maxPositionEmbeddings / originalMaxPositionEmbeddings;
    if (factor < 1.0) return 1.0;
    return math.sqrt(
      1.0 +
          math.log(factor) /
              math.log(originalMaxPositionEmbeddings.toDouble()),
    );
  }

  @override
  MLXArray call(MLXArray x, int offset) {
    MLXArray input = x;
    if (_scale != 1.0) {
      final rank = x.ndim;
      final headDim = x.dim(rank - 1);
      if (dims < headDim) {
        // Scale only the first [dims] channels on the last axis, then put back.
        final stop = List<int>.from(x.shape);
        stop[rank - 1] = dims;
        final xFirst = x.slice(start: List.filled(rank, 0), stop: stop);
        final scaleArr = MLXArray.float_(x.context, _scale);
        final xScaled = xFirst * scaleArr;
        xFirst.dispose();
        scaleArr.dispose();
        input = x.sliceUpdate(xScaled,
            start: List.filled(rank, 0), stop: stop);
        xScaled.dispose();
      } else {
        final scaleArr = MLXArray.float_(x.context, _scale);
        input = x * scaleArr;
        scaleArr.dispose();
      }
    }
    final result = input.rope(
      dims: dims,
      traditional: traditional,
      offset: offset,
      freqs: _freqs,
    );
    if (!identical(input, x)) input.dispose();
    return result;
  }

  @override
  void dispose() => _freqs.dispose();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Creates the appropriate [RopeLayer] from a model's rope configuration.
///
/// Reads `rope_scaling['type']` (or `rope_type`) from [scalingConfig] and
/// returns the matching implementation:
/// - `"default"` / `"linear"` → [DefaultRope]
/// - `"llama3"` → [Llama3Rope]
/// - `"yarn"` / `"deepseek_yarn"` / `"telechat3-yarn"` → [YarnRope]
/// - `"mrope"` → [DefaultRope] (multimodal rotation is handled by the model)
/// - unknown / null → [DefaultRope]
///
/// Mirrors `initializeRope()` from mlx-swift-lm.
RopeLayer initializeRope(
  MLXContext ctx, {
  required int dims,
  required double base,
  bool traditional = false,
  Map<String, dynamic>? scalingConfig,
  int maxPositionEmbeddings = 2048,
}) {
  final ropeType = scalingConfig?['type'] as String? ??
      scalingConfig?['rope_type'] as String? ??
      'default';

  if (ropeType == 'default' || ropeType == 'linear') {
    final scale = ropeType == 'linear'
        ? 1.0 / ((scalingConfig?['factor'] as num?)?.toDouble() ?? 1.0)
        : 1.0;
    return DefaultRope(dims: dims, traditional: traditional, base: base, scale: scale);
  }

  if (ropeType == 'llama3') {
    return Llama3Rope(
      ctx,
      dims: dims,
      traditional: traditional,
      base: base,
      scalingConfig: scalingConfig!,
    );
  }

  const yarnTypes = {'yarn', 'deepseek_yarn', 'telechat3-yarn'};
  if (yarnTypes.contains(ropeType)) {
    return YarnRope(
      ctx,
      dims: dims,
      traditional: traditional,
      base: base,
      scalingFactor: (scalingConfig?['factor'] as num?)?.toDouble() ?? 32.0,
      originalMaxPositionEmbeddings:
          (scalingConfig?['original_max_position_embeddings'] as num?)?.toInt() ?? 4096,
      betaFast: (scalingConfig?['beta_fast'] as num?)?.toDouble() ?? 32.0,
      betaSlow: (scalingConfig?['beta_slow'] as num?)?.toDouble() ?? 1.0,
      mscale: (scalingConfig?['mscale'] as num?)?.toDouble() ?? 1.0,
      mscaleAllDim: (scalingConfig?['mscale_all_dim'] as num?)?.toDouble() ?? 0.0,
    );
  }

  if (ropeType == 'longrope' || ropeType == 'su') {
    final longFactor = (scalingConfig?['long_factor'] as List?)
            ?.map((v) => (v as num).toDouble())
            .toList() ??
        <double>[];
    final origMaxPos =
        (scalingConfig?['original_max_position_embeddings'] as num?)?.toInt() ??
            maxPositionEmbeddings;
    final longMScale = (scalingConfig?['long_mscale'] as num?)?.toDouble();
    return SuScaledRope(
      ctx,
      dims: dims,
      traditional: traditional,
      base: base,
      longFactor: longFactor,
      maxPositionEmbeddings: maxPositionEmbeddings,
      originalMaxPositionEmbeddings: origMaxPos,
      longMScale: longMScale,
    );
  }

  // "mrope": multi-modal rotation is handled in the attention layer.
  return DefaultRope(dims: dims, traditional: traditional, base: base);
}
