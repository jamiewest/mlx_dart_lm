import 'dart:async';

import 'package:mlx_dart/mlx_dart.dart' hide KVCache;

import 'evaluate.dart';
import 'generate_parameters.dart';
import 'kv_cache.dart';
import 'language_model.dart';
import 'tokenizer.dart';

// ---------------------------------------------------------------------------
// Streaming generation
// ---------------------------------------------------------------------------

/// A single chunk of decoded text emitted during [generateStream].
final class GenerateChunk {
  const GenerateChunk(this.text);
  final String text;
}

/// Timing summary emitted as the final event of [generateStream].
final class GenerateInfo {
  const GenerateInfo({
    required this.promptTokenCount,
    required this.generationTokenCount,
    required this.promptDuration,
    required this.generationDuration,
    required this.stopReason,
  });

  final int promptTokenCount;
  final int generationTokenCount;
  final Duration promptDuration;
  final Duration generationDuration;
  final GenerateStopReason stopReason;

  double get promptTokensPerSecond =>
      promptDuration.inMicroseconds > 0
          ? promptTokenCount / promptDuration.inMicroseconds * 1e6
          : 0;

  double get tokensPerSecond =>
      generationDuration.inMicroseconds > 0
          ? generationTokenCount / generationDuration.inMicroseconds * 1e6
          : 0;

  @override
  String toString() =>
      'Prompt: $promptTokenCount tokens, '
      '${promptTokensPerSecond.toStringAsFixed(1)} tok/s | '
      'Generation: $generationTokenCount tokens, '
      '${tokensPerSecond.toStringAsFixed(1)} tok/s';
}

/// Why [generateStream] stopped producing tokens.
enum GenerateStopReason { eosToken, maxTokens, cancelled }

/// Streams decoded text chunks from [model] given [promptTokens].
///
/// Yields [GenerateChunk] events for each decoded piece of text, then a
/// final [GenerateInfo] event with timing statistics.
///
/// Cancel the returned [StreamSubscription] to stop generation early
/// (the final event will carry [GenerateStopReason.cancelled]).
///
/// Mirrors `generateStream` from mlx-swift-lm.
Stream<Object> generateStream({
  required MLXContext ctx,
  required LanguageModel model,
  required List<int> promptTokens,
  required Tokenizer tokenizer,
  List<KVCache>? cache,
  GenerateParameters? parameters,
  LogitSampler? sampler,
  LogitProcessor? processor,
}) async* {
  final params = parameters ?? const GenerateParameters();
  final eosTokenId = tokenizer.eosTokenId;

  final promptStart = DateTime.now();
  final input = LMInput.tokens(MLXArray.fromInts(ctx, promptTokens));

  final iterator = TokenIterator.fromInput(
    input: input,
    model: model,
    ctx: ctx,
    cache: cache,
    parameters: params,
    sampler: sampler,
    processor: processor,
  );

  final promptDuration = DateTime.now().difference(promptStart);

  final detokenizer = StreamingDetokenizer(tokenizer);
  var stopReason = GenerateStopReason.maxTokens;
  final genStart = DateTime.now();

  while (iterator.moveNext()) {
    final token = iterator.current;

    if (eosTokenId != null && token == eosTokenId) {
      stopReason = GenerateStopReason.eosToken;
      break;
    }

    final chunk = detokenizer.append(token);
    if (chunk != null && chunk.isNotEmpty) {
      yield GenerateChunk(chunk);
    }
  }

  yield GenerateInfo(
    promptTokenCount: promptTokens.length,
    generationTokenCount: iterator.tokenCount,
    promptDuration: promptDuration,
    generationDuration: DateTime.now().difference(genStart),
    stopReason: stopReason,
  );
}
