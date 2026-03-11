import 'package:mlx_lm_common/mlx_lm_common.dart';
import 'package:mlx_dart/mlx_dart.dart';

import 'models/cohere.dart';
import 'models/deepseek_v3.dart';
import 'models/gemma.dart';
import 'models/gemma2.dart';
import 'models/gemma3_text.dart';
import 'models/internlm2.dart';
import 'models/llama.dart';
import 'models/minicpm.dart';
import 'models/mixtral.dart';
import 'models/phi.dart';
import 'models/phi3.dart';
import 'models/olmoe.dart';
import 'models/olmo2.dart';
import 'models/openelm.dart';
import 'models/phimoe.dart';
import 'models/qwen3_moe.dart';
import 'models/starcoder2.dart';

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// Creates a [LanguageModel] from a JSON config map.
///
/// The [modelType] field in the config (e.g. `'llama'`, `'mistral'`) is used
/// to select the right architecture. Throws [UnknownModelTypeException] for
/// unrecognised types.
///
/// Mirrors the model-type registry in mlx-swift-lm.
LanguageModel modelFromConfig(
  MLXContext ctx,
  Map<String, dynamic> config,
) {
  final type = (config['model_type'] as String? ?? '').toLowerCase();
  return switch (type) {
    'llama' || 'mistral' || 'qwen2' || 'granite' => LlamaModel(
        ctx,
        LlamaConfig.fromJson(config),
      ),
    'gemma' => GemmaModel(ctx, GemmaConfig.fromJson(config)),
    'gemma2' => Gemma2Model(ctx, Gemma2Config.fromJson(config)),
    'gemma3_text' => Gemma3TextModel(ctx, Gemma3TextConfig.fromJson(config)),
    'phi3' => Phi3Model(ctx, Phi3Config.fromJson(config)),
    'cohere' || 'cohere2' => CohereModel(ctx, CohereConfig.fromJson(config)),
    'qwen3' => LlamaModel(ctx, LlamaConfig.fromJson(config)),
    'starcoder2' => Starcoder2Model(ctx, Starcoder2Config.fromJson(config)),
    'internlm2' => Internlm2Model(ctx, Internlm2Config.fromJson(config)),
    'minicpm' => MiniCPMModel(ctx, MiniCPMConfig.fromJson(config)),
    'qwen3_moe' => Qwen3MoEModel(ctx, Qwen3MoEConfig.fromJson(config)),
    'phi' => PhiModel(ctx, PhiConfig.fromJson(config)),
    'deepseek_v3' => DeepSeekV3Model(ctx, DeepSeekV3Config.fromJson(config)),
    'mixtral' => MixtralModel(ctx, MixtralConfig.fromJson(config)),
    'olmoe' => OlmoEModel(ctx, OlmoEConfig.fromJson(config)),
    'olmo2' => Olmo2Model(ctx, Olmo2Config.fromJson(config)),
    'openelm' => OpenELMModel(ctx, OpenELMConfig.fromJson(config)),
    'phimoe' => PhiMoEModel(ctx, PhiMoEConfig.fromJson(config)),
    _ => throw UnknownModelTypeException(type),
  };
}

/// Thrown when [modelFromConfig] encounters an unregistered model type.
final class UnknownModelTypeException implements Exception {
  const UnknownModelTypeException(this.modelType);
  final String modelType;

  @override
  String toString() => 'UnknownModelTypeException: unsupported model_type "$modelType"';
}
