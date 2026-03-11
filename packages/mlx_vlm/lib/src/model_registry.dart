import 'package:mlx_dart/mlx_dart.dart';

import 'models/lfm2_vl.dart';
import 'models/qwen2_vl.dart';
import 'vlm_model.dart';

// ---------------------------------------------------------------------------
// VLM model registry
// ---------------------------------------------------------------------------

/// Creates a [VLMModel] from a JSON config map.
///
/// The `model_type` field in the config selects the architecture.
/// Throws [UnknownVLMTypeException] for unrecognised types.
VLMModel vlmFromConfig(
  MLXContext ctx,
  Map<String, dynamic> config,
) {
  final type = (config['model_type'] as String? ?? '').toLowerCase();
  return switch (type) {
    'qwen2_vl' || 'qwen2vl' => Qwen2VLModel(
        ctx,
        Qwen2VLConfig.fromJson(config),
      ),
    'lfm2_vl' || 'lfm2vl' || 'lfm2-vl' => Lfm2VLModel(
        ctx,
        Lfm2Config.fromJson(config),
      ),
    _ => throw UnknownVLMTypeException(type),
  };
}

/// Thrown when [vlmFromConfig] encounters an unregistered model type.
final class UnknownVLMTypeException implements Exception {
  const UnknownVLMTypeException(this.modelType);
  final String modelType;

  @override
  String toString() =>
      'UnknownVLMTypeException: unsupported model_type "$modelType"';
}
