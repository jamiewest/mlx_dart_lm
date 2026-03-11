import 'package:mlx_dart/mlx_dart.dart' hide KVCache;

import 'generate_parameters.dart';
import 'kv_cache.dart';

/// Time/Height/Width dimensions for video/image inputs.
final class THW {
  const THW(this.t, this.h, this.w);

  final int t;
  final int h;
  final int w;

  int get product => t * h * w;

  @override
  String toString() => 'THW($t, $h, $w)';
}

/// Tokenized text input with an optional attention mask.
final class LMInputText {
  const LMInputText({required this.tokens, this.mask});

  /// Input token array — shape `[batch, seqLen]`.
  final MLXArray tokens;

  /// Optional attention mask — shape `[batch, seqLen]`.
  final MLXArray? mask;

  /// Slice the tokens (and mask, if present) along the sequence axis.
  LMInputText slice({required List<int> start, required List<int> stop}) {
    final rank = tokens.ndim;
    final strides = List.filled(rank, 1);
    return LMInputText(
      tokens: tokens.slice(start: start, stop: stop, strides: strides),
      mask: mask?.slice(start: start, stop: stop, strides: strides),
    );
  }
}

/// Prepared image input for a VLM.
final class ProcessedImage {
  const ProcessedImage({required this.pixels, this.frames});

  final MLXArray pixels;
  final List<THW>? frames;
}

/// Prepared video input for a VLM.
final class ProcessedVideo {
  const ProcessedVideo({required this.pixels, this.frames});

  final MLXArray pixels;
  final List<THW>? frames;
}

/// Full input to a [LanguageModel]. Text is always present; image/video are
/// optional for multi-modal models.
///
/// Mirrors `LMInput` from mlx-swift-lm.
final class LMInput {
  const LMInput({
    required this.text,
    this.image,
    this.video,
  });

  LMInput.tokens(MLXArray tokens, {MLXArray? mask})
      : this(text: LMInputText(tokens: tokens, mask: mask));

  final LMInputText text;
  final ProcessedImage? image;
  final ProcessedVideo? video;
}

/// Per-step output from a [LanguageModel].
///
/// Mirrors `LMOutput` from mlx-swift-lm.
final class LMOutput {
  const LMOutput({required this.logits, this.state});

  /// Unnormalised token log-probabilities — shape `[batch, seqLen, vocabSize]`.
  final MLXArray logits;

  /// Optional recurrent state carried into the next step (e.g. cross-attention).
  final LMState? state;
}

/// Optional recurrent state passed between generation steps.
final class LMState {
  const LMState({this.crossAttentionStates});

  final MLXArray? crossAttentionStates;
}

/// The result of [LanguageModel.prepare].
///
/// Mirrors `PrepareResult` from mlx-swift-lm.
sealed class PrepareResult {
  const PrepareResult();

  /// Return these remaining tokens to the [TokenIterator].
  const factory PrepareResult.tokens(LMInputText tokens) = TokensPrepareResult;

  /// A logit array to use for the first token (prompt already processed).
  const factory PrepareResult.logits(LMOutput output) = LogitsPrepareResult;
}

/// [PrepareResult] carrying remaining tokens for the [TokenIterator].
final class TokensPrepareResult extends PrepareResult {
  const TokensPrepareResult(this.tokens);
  final LMInputText tokens;
}

/// [PrepareResult] carrying pre-computed logits (prompt fully consumed).
final class LogitsPrepareResult extends PrepareResult {
  const LogitsPrepareResult(this.output);
  final LMOutput output;
}

/// Core protocol for all language models (LLM and VLM).
///
/// Mirrors the `LanguageModel` protocol from mlx-swift-lm.
abstract interface class LanguageModel {
  /// Pre-fill the KV cache from [input] and return what the [TokenIterator]
  /// should process next.
  PrepareResult prepare(LMInput input, List<KVCache> cache, {int? windowSize});

  /// Compute one generation step for [input] given [cache] and optional [state].
  LMOutput call(LMInputText input, {List<KVCache>? cache, LMState? state});

  /// Create a fresh set of KV caches for generation.
  List<KVCache> newCache(GenerateParameters? parameters);

  /// Sanitise a raw weight map, e.g. to rename or remove keys.
  ///
  /// Called by the weight loader before assigning tensors to the model.
  Map<String, MLXArray> sanitizeWeights(Map<String, MLXArray> weights) => weights;
}

/// Optional protocol for models that can provide KV head counts per layer.
///
/// When implemented, [newCache] is provided automatically.
abstract interface class KVCacheDimensionProvider implements LanguageModel {
  List<int> get kvHeads;
}

