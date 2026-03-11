import 'dart:math' as math;

import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart';

import 'vlm_model.dart';

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

/// Normalize an image array element-wise: `(x - mean) / std`.
///
/// [image] must be `[H, W, C]` or `[B, H, W, C]` float32 in `[0, 1]`.
/// [norm] provides per-channel mean and standard deviation.
MLXArray normalizeImage(MLXContext ctx, MLXArray image, ImageNorm norm) {
  final c = norm.mean.length;
  final rank = image.ndim;
  final shape = List.filled(rank, 1)..[rank - 1] = c;
  final mean = MLXArray.fromFloats(ctx, norm.mean, shape: shape);
  final std = MLXArray.fromFloats(ctx, norm.std, shape: shape);
  final centered = image - mean;
  mean.dispose();
  final result = centered / std;
  centered.dispose();
  std.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// SigLIP image processor
// ---------------------------------------------------------------------------
//
// SigLIP (and therefore LFM2.4 Vision) expects square images normalised to
// the range [-1, 1] with mean = std = 0.5.
//
// The caller is responsible for resizing [pixels] to
// [imageSize × imageSize] before calling [process] (e.g. via
// [bilinearResize]).  [process] handles only normalisation and batching.

/// [ImageProcessor] for SigLIP-based vision encoders (e.g. LFM2.4 Vision).
///
/// Normalises a `[imageSize, imageSize, C]` float32 image and returns a
/// `ProcessedImage` with pixels `[1, imageSize, imageSize, C]`.
final class SigLIPImageProcessor implements ImageProcessor {
  const SigLIPImageProcessor({
    required this.imageSize,
    this.norm = ImageNorm.siglip,
  });

  final int imageSize;
  final ImageNorm norm;

  /// Process [pixels] — shape `[H, W, C]` float32 in `[0, 1]`.
  ///
  /// [pixels] is bilinear-resized to `[imageSize × imageSize]` when the
  /// dimensions do not already match, then normalised.
  @override
  ProcessedImage process(MLXContext ctx, MLXArray pixels) {
    final h = pixels.dim(0);
    final w = pixels.dim(1);

    final resized = (h == imageSize && w == imageSize)
        ? pixels
        : bilinearResize(ctx, pixels, imageSize, imageSize);

    final normed = normalizeImage(ctx, resized, norm);
    if (!identical(resized, pixels)) resized.dispose();

    final batched = normed.expandDims(0); // [1, imageSize, imageSize, C]
    normed.dispose();
    return ProcessedImage(pixels: batched);
  }
}

// ---------------------------------------------------------------------------
// Qwen2-VL image processor
// ---------------------------------------------------------------------------
//
// Qwen2-VL uses a dynamic resolution approach: the image is resized so that
// H and W are multiples of [patchSize] × [mergeSize], then patchified into
// a flat patch sequence `[numPatches, temporalPatch × C × pH × pW]`.
//
// For a single static image:
//   temporalPatch = 1
//   numPatches = gridH × gridW
//   gridH = H / patchSize, gridW = W / patchSize
//
// The patch tensor is compatible with the Qwen2-VL vision encoder's
// [_Qwen2VLVisionTransformer.forwardWithThw] input.

/// [ImageProcessor] for Qwen2-VL vision models.
///
/// Resizes, normalises, and patchifies a single image into the flat patch
/// layout expected by the Qwen2-VL vision transformer.
final class Qwen2VLImageProcessor implements ImageProcessor {
  const Qwen2VLImageProcessor({
    required this.patchSize,
    required this.temporalPatchSize,
    required this.mergeSize,
    this.minPixels = 256 * 28 * 28,
    this.maxPixels = 1280 * 28 * 28,
    this.norm = ImageNorm.clip,
  });

  final int patchSize;
  final int temporalPatchSize;
  final int mergeSize;

  /// Minimum total pixel budget (before patching).
  final int minPixels;

  /// Maximum total pixel budget (before patching).
  final int maxPixels;

  final ImageNorm norm;

  /// The spatial downsampling factor applied by the merger: [patchSize] ×
  /// [mergeSize].
  int get spatialFactor => patchSize * mergeSize;

  /// Process [pixels] — shape `[H, W, C]` float32 in `[0, 1]`.
  ///
  /// Returns a `ProcessedImage` with:
  /// - `pixels`: `[numPatches, temporalPatch × C × pH × pW]`
  /// - `frames`: one [THW] entry describing the grid layout.
  @override
  ProcessedImage process(MLXContext ctx, MLXArray pixels) {
    final h = pixels.dim(0);
    final w = pixels.dim(1);
    final c = pixels.dim(2);

    // Snap to spatial factor multiples that respect the pixel budget.
    final (targetH, targetW) = _smartResize(h, w);

    final resized = (h == targetH && w == targetW)
        ? pixels
        : bilinearResize(ctx, pixels, targetH, targetW);

    final normed = normalizeImage(ctx, resized, norm);
    if (!identical(resized, pixels)) resized.dispose();

    final gridH = targetH ~/ patchSize;
    final gridW = targetW ~/ patchSize;
    const t = 1; // single frame (static image)

    // Reshape: [H, W, C] → [gridH, pH, gridW, pW, C]
    final r1 = normed.reshape([gridH, patchSize, gridW, patchSize, c]);
    normed.dispose();

    // Permute: [gridH, gridW, pH, pW, C] → later flattened to patches
    final r2 = r1.transpose([0, 2, 1, 3, 4]); // [gridH, gridW, pH, pW, C]
    r1.dispose();

    // Flat patches: [gridH*gridW, pH*pW*C]
    final patchDim = patchSize * patchSize * c;
    final flat = r2.reshape([gridH * gridW, patchDim]);
    r2.dispose();

    return ProcessedImage(
      pixels: flat,
      frames: [THW(t, gridH, gridW)],
    );
  }

  // Smart resize: find (targetH, targetW) that are multiples of [spatialFactor]
  // with aspect ratio close to the original and total pixels within budget.
  (int, int) _smartResize(int h, int w) {
    final factor = spatialFactor;

    // Scale so total pixels ≈ max(min(h*w, maxPixels), minPixels)
    final currentPixels = h * w;
    double scale = 1.0;
    if (currentPixels > maxPixels) {
      scale = math.sqrt(maxPixels / currentPixels);
    } else if (currentPixels < minPixels) {
      scale = math.sqrt(minPixels / currentPixels);
    }

    var th = (h * scale).round();
    var tw = (w * scale).round();

    // Round to nearest multiple of spatial factor.
    th = ((th + factor ~/ 2) ~/ factor) * factor;
    tw = ((tw + factor ~/ 2) ~/ factor) * factor;

    // Ensure at least one patch on each side.
    th = th < factor ? factor : th;
    tw = tw < factor ? factor : tw;

    return (th, tw);
  }
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/// Returns the [SigLIPImageProcessor] for a LFM2.4 Vision config JSON.
SigLIPImageProcessor sigLIPProcessorFromConfig(Map<String, dynamic> json) {
  final visionJson = (json['vision_config'] as Map<String, dynamic>?) ?? json;
  final imageSize = (visionJson['image_size'] as num?)?.toInt() ?? 384;
  return SigLIPImageProcessor(imageSize: imageSize);
}

/// Returns the [Qwen2VLImageProcessor] for a Qwen2-VL config JSON.
Qwen2VLImageProcessor qwen2VLProcessorFromConfig(Map<String, dynamic> json) {
  final vj = json['vision_config'] as Map<String, dynamic>?;
  final patchSize = (vj?['patch_size'] as num?)?.toInt() ??
      (json['patch_size'] as num?)?.toInt() ??
      14;
  final temporalPatch = (vj?['temporal_patch_size'] as num?)?.toInt() ??
      (json['temporal_patch_size'] as num?)?.toInt() ??
      2;
  final mergeSize = (vj?['spatial_merge_size'] as num?)?.toInt() ??
      (json['spatial_merge_size'] as num?)?.toInt() ??
      2;
  return Qwen2VLImageProcessor(
    patchSize: patchSize,
    temporalPatchSize: temporalPatch,
    mergeSize: mergeSize,
  );
}

/// Create the appropriate [ImageProcessor] from a VLM config JSON.
///
/// Dispatches on the `model_type` field, same as [vlmFromConfig].
ImageProcessor imageProcessorFromConfig(Map<String, dynamic> json) {
  final type = (json['model_type'] as String? ?? '').toLowerCase();
  return switch (type) {
    'qwen2_vl' || 'qwen2vl' => qwen2VLProcessorFromConfig(json),
    'lfm2_vl' || 'lfm2vl' || 'lfm2-vl' => sigLIPProcessorFromConfig(json),
    _ => throw ArgumentError('No image processor registered for model_type "$type"'),
  };
}
