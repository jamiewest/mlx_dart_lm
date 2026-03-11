import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart' hide KVCache, RotatingKVCache;

// ---------------------------------------------------------------------------
// VisionModel — encodes image patches into hidden states
// ---------------------------------------------------------------------------

/// Encodes a pre-processed image into feature embeddings.
///
/// Input pixels must be prepared by a matching [ImageProcessor] before being
/// passed here.  The output shape varies by architecture but is typically
/// `[batch, numPatches, embedDim]`.
abstract interface class VisionModel {
  MLXArray encodeImage(MLXContext ctx, MLXArray pixels);
}

// ---------------------------------------------------------------------------
// VLMModel — LanguageModel that can also handle image inputs
// ---------------------------------------------------------------------------

/// A language model that accepts multimodal (text + image/video) inputs.
///
/// Extends [LanguageModel] with the ability to merge visual feature embeddings
/// into the token embedding space before autoregressive decoding.
///
/// Mirrors the `VLMModel` protocol from mlx-swift-lm.
abstract interface class VLMModel implements LanguageModel {
  /// Encode and merge [input]'s image/video into text embeddings, then run
  /// a forward pass.  Called by [prepare] for chunked prefill.
  LMOutput callWithPixels(
    MLXContext ctx,
    LMInput input, {
    List<KVCache>? cache,
    LMState? state,
  });

  /// The vision component of this model.
  VisionModel get visionModel;
}

// ---------------------------------------------------------------------------
// ImageProcessor
// ---------------------------------------------------------------------------

/// Normalisation constants for a vision encoder.
final class ImageNorm {
  const ImageNorm({required this.mean, required this.std});

  final List<double> mean; // per-channel
  final List<double> std;  // per-channel

  static const siglip = ImageNorm(
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  );

  static const clip = ImageNorm(
    mean: [0.48145466, 0.4578275, 0.40821073],
    std: [0.26862954, 0.26130258, 0.27577711],
  );

  static const imagenet = ImageNorm(
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  );
}

/// Options for resizing and tiling an image before encoding.
final class ImageProcessorConfig {
  const ImageProcessorConfig({
    required this.imageSize,
    required this.norm,
    this.patchSize = 14,
    this.temporalPatchSize = 1,
    this.spatialMergeSize = 1,
  });

  final int imageSize;
  final ImageNorm norm;
  final int patchSize;
  final int temporalPatchSize;
  final int spatialMergeSize;
}

/// Converts raw pixel data into a [ProcessedImage] ready for [VisionModel].
///
/// Implementations handle resize, normalise, and patchify.
abstract interface class ImageProcessor {
  /// Process [pixels] — shape `[H, W, C]` float32 in [0,1] — into
  /// a [ProcessedImage] with the correct patch layout.
  ProcessedImage process(MLXContext ctx, MLXArray pixels);
}
