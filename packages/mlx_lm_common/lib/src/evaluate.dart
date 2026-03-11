import 'dart:typed_data';

import 'package:mlx_dart/mlx_dart.dart' hide KVCache;

import 'generate_parameters.dart';
import 'kv_cache.dart';
import 'language_model.dart';

// ---------------------------------------------------------------------------
// Logit samplers
// ---------------------------------------------------------------------------

/// Converts a logit vector into a single sampled token ID.
///
/// Mirrors `LogitSampler` from mlx-swift-lm.
abstract interface class LogitSampler {
  MLXArray sample(MLXContext ctx, MLXArray logits);
}

/// Greedy (argmax) sampler — always picks the most likely token.
final class ArgMaxSampler implements LogitSampler {
  const ArgMaxSampler();

  @override
  MLXArray sample(MLXContext ctx, MLXArray logits) => logits.argmax(axis: -1);
}

/// Temperature-scaled categorical sampler.
final class CategoricalSampler implements LogitSampler {
  const CategoricalSampler({this.temperature = 0.6});

  final double temperature;

  @override
  MLXArray sample(MLXContext ctx, MLXArray logits) =>
      categoricalSample(ctx, logits, temperature);
}

/// Top-p (nucleus) sampler with temperature scaling.
///
/// Filters the vocabulary to tokens whose cumulative probability >= [topP]
/// before sampling.
final class TopPSampler implements LogitSampler {
  const TopPSampler({required this.temperature, required this.topP});

  final double temperature;
  final double topP;

  @override
  MLXArray sample(MLXContext ctx, MLXArray logits) {
    // Work in float32 for numerical stability.
    final l = logits.dtype == MLXDtype.bfloat16
        ? logits.astype(MLXDtype.float32)
        : logits;
    final ownL = !identical(l, logits);

    final temp = MLXArray.float_(ctx, temperature);
    final scaled = l / temp;
    temp.dispose();
    if (ownL) l.dispose();

    // Compute probabilities.
    final probs = scaled.softmax(axis: -1);
    scaled.dispose();

    // Sort indices in descending probability order.
    final sortedIdx = probs.argsort(axis: -1); // ascending
    // Reverse to get descending: flip indices along last axis
    final rank = sortedIdx.ndim;
    final len = sortedIdx.dim(-1);
    final flipIdx = MLXArray.arange(ctx, (len - 1).toDouble(), -1.0, -1.0, dtype: MLXDtype.int32);
    final descIdx = sortedIdx.take(flipIdx, axis: rank - 1);
    flipIdx.dispose();
    sortedIdx.dispose();

    final sortedProbs = probs.take(descIdx, axis: rank - 1);
    probs.dispose();

    // Cumulative sum and top-p mask.
    final cumProbs = sortedProbs.cumsum(axis: -1);
    final threshold = MLXArray.float_(ctx, topP);
    final mask = cumProbs.less(threshold); // keep tokens before cumsum exceeds topP
    threshold.dispose();
    cumProbs.dispose();

    final zeros = MLXArray.zeros(ctx, sortedProbs.shape);
    final filtered = where(ctx, mask, sortedProbs, zeros);
    sortedProbs.dispose();
    zeros.dispose();
    mask.dispose();

    // Sample from filtered distribution (re-normalisation handled by categorical).
    final filteredLog = filtered.log();
    filtered.dispose();
    final sampledPos = categoricalSample(ctx, filteredLog, 1.0);
    filteredLog.dispose();

    // Map back from sorted position to original vocabulary index.
    final result = descIdx.take(sampledPos, axis: rank - 1);
    descIdx.dispose();
    sampledPos.dispose();
    return result;
  }
}

// ---------------------------------------------------------------------------
// Logit processors
// ---------------------------------------------------------------------------

/// Visits and optionally modifies logits at each generation step.
///
/// Mirrors `LogitProcessor` from mlx-swift-lm.
abstract interface class LogitProcessor {
  void onPrompt(MLXContext ctx, MLXArray promptTokens);
  MLXArray process(MLXContext ctx, MLXArray logits);
  void didSample(int token);
}

/// Penalises recently-seen tokens to discourage repetition.
///
/// Mirrors `RepetitionContext` from mlx-swift-lm.
final class RepetitionPenaltyProcessor implements LogitProcessor {
  RepetitionPenaltyProcessor({
    required this.penalty,
    required this.contextSize,
  }) : assert(contextSize > 0);

  final double penalty;
  final int contextSize;

  final List<int> _tokens = [];
  int _idx = 0;

  @override
  void onPrompt(MLXContext ctx, MLXArray promptTokens) {
    promptTokens.eval();
    ctx.synchronize();
    final data = promptTokens.toInt32List();
    _tokens.clear();
    final start = data.length > contextSize ? data.length - contextSize : 0;
    for (var i = start; i < data.length; i++) {
      _tokens.add(data[i]);
    }
    _idx = 0;
  }

  @override
  MLXArray process(MLXContext ctx, MLXArray logits) {
    if (_tokens.isEmpty) return logits;

    // Pull logits to CPU in float32, apply penalty, return new array.
    // This is a per-token (not per-prefill) operation on a 1-D vocabSize vector,
    // so the CPU round-trip cost is negligible.
    final l = logits.dtype != MLXDtype.float32
        ? logits.astype(MLXDtype.float32)
        : logits;
    l.eval();
    ctx.synchronize();
    final data = Float32List.fromList(l.toFloat32List());
    if (!identical(l, logits)) l.dispose();

    for (final idx in _tokens) {
      if (idx < 0 || idx >= data.length) continue;
      final v = data[idx];
      data[idx] = v < 0 ? v * penalty : v / penalty;
    }

    return MLXArray.fromFloats(ctx, data, dtype: MLXDtype.float32);
  }

  @override
  void didSample(int token) {
    if (_tokens.length < contextSize) {
      _tokens.add(token);
    } else {
      _tokens[_idx] = token;
      _idx = (_idx + 1) % contextSize;
    }
  }
}

// ---------------------------------------------------------------------------
// GenerateParameters helpers
// ---------------------------------------------------------------------------

LogitSampler _buildSampler(GenerateParameters p) {
  if (p.temperature == 0) return const ArgMaxSampler();
  if (p.topP > 0 && p.topP < 1) {
    return TopPSampler(temperature: p.temperature, topP: p.topP);
  }
  return CategoricalSampler(temperature: p.temperature);
}

LogitProcessor? _buildProcessor(GenerateParameters p) {
  if (p.repetitionPenalty case final penalty? when p.repetitionContextSize > 0) {
    return RepetitionPenaltyProcessor(
        penalty: penalty, contextSize: p.repetitionContextSize);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Token iterator
// ---------------------------------------------------------------------------

/// Iterates over generated token IDs, one per [moveNext] call.
///
/// Use the higher-level [generateStream] function for streaming text output.
/// Mirrors `TokenIterator` from mlx-swift-lm.
final class TokenIterator implements Iterator<int> {
  TokenIterator._({
    required this.model,
    required this.ctx,
    required LMInputText initialText,
    required List<KVCache> cache,
    required this.sampler,
    this.processor,
    this.maxTokens,
  })  : _y = initialText,
        _cache = cache;

  final LanguageModel model;
  final MLXContext ctx;
  final LogitSampler sampler;
  final LogitProcessor? processor;
  final int? maxTokens;

  LMInputText _y;
  final List<KVCache> _cache;
  LMState? _state;
  int _tokenCount = 0;
  int _current = -1;
  bool _done = false;

  /// Creates a [TokenIterator] by pre-filling the model cache from [input].
  factory TokenIterator.fromInput({
    required LMInput input,
    required LanguageModel model,
    required MLXContext ctx,
    List<KVCache>? cache,
    GenerateParameters? parameters,
    LogitSampler? sampler,
    LogitProcessor? processor,
  }) {
    final params = parameters ?? const GenerateParameters();
    final resolvedCache = cache ?? model.newCache(params);
    final resolvedSampler = sampler ?? _buildSampler(params);
    final resolvedProcessor = processor ?? _buildProcessor(params);

    final it = TokenIterator._(
      model: model,
      ctx: ctx,
      initialText: input.text,
      cache: resolvedCache,
      sampler: resolvedSampler,
      processor: resolvedProcessor,
      maxTokens: params.maxTokens,
    );
    it._prefill(input, params.prefillStepSize);
    return it;
  }

  void _prefill(LMInput input, int stepSize) {
    processor?.onPrompt(ctx, input.text.tokens);

    switch (model.prepare(input, _cache, windowSize: stepSize)) {
      case TokensPrepareResult(:final tokens):
        _y = tokens;
        final first = _step(_y);
        _y = LMInputText(tokens: first);
        first.eval();

      case LogitsPrepareResult(:final output):
        final t = _convertToToken(output.logits);
        _y = LMInputText(tokens: t);
        _state = output.state;
        t.eval();
    }
  }

  MLXArray _step(LMInputText prev) {
    final expanded = LMInputText(
      tokens: prev.tokens.expandDims(0),
      mask: prev.mask?.expandDims(0),
    );
    final output = model.call(expanded, cache: _cache, state: _state);
    _state = output.state;
    expanded.tokens.dispose();
    expanded.mask?.dispose();
    return _convertToToken(output.logits);
  }

  MLXArray _convertToToken(MLXArray logits) {
    // logits shape: [batch, seqLen, vocabSize] → take last position → [batch, vocabSize]
    final rank = logits.ndim;
    final seqLen = logits.dim(rank - 2);
    final start = List.filled(rank, 0)..[rank - 2] = seqLen - 1;
    final stop = logits.shape..[rank - 2] = seqLen;
    var l = logits.slice(start: start, stop: stop).squeeze(axis: rank - 2);

    if (processor != null) {
      final processed = processor!.process(ctx, l);
      if (!identical(processed, l)) l.dispose();
      l = processed;
    }

    final t = sampler.sample(ctx, l);
    if (!identical(l, logits)) l.dispose();
    processor?.didSample(t.itemInt());
    return t;
  }

  @override
  bool moveNext() {
    if (_done) return false;
    if (maxTokens != null && _tokenCount >= maxTokens!) {
      _done = true;
      return false;
    }

    _current = _y.tokens.itemInt();

    final next = _step(_y);
    _y = LMInputText(tokens: next);
    next.eval();

    _tokenCount++;
    return true;
  }

  @override
  int get current => _current;

  int get tokenCount => _tokenCount;
}

