/// Parameters controlling text generation.
///
/// Mirrors `GenerateParameters` from mlx-swift-lm.
final class GenerateParameters {
  const GenerateParameters({
    this.maxTokens,
    this.maxKVSize,
    this.kvBits,
    this.kvGroupSize = 64,
    this.quantizedKVStart = 0,
    this.temperature = 0.6,
    this.topP = 1.0,
    this.repetitionPenalty,
    this.repetitionContextSize = 20,
    this.prefillStepSize = 512,
  });

  final int? maxTokens;

  /// Maximum number of tokens in the KV cache. When set, uses [RotatingKVCache].
  final int? maxKVSize;

  /// Number of bits for KV cache quantization. `null` disables quantization.
  final int? kvBits;

  /// Group size for KV cache quantization (default: 64).
  final int kvGroupSize;

  /// Token step at which quantized KV cache kicks in (default: 0).
  final int quantizedKVStart;

  final double temperature;
  final double topP;
  final double? repetitionPenalty;
  final int repetitionContextSize;
  final int prefillStepSize;
}
