/// Abstract tokenizer interface used across the LM framework.
///
/// Mirrors the `Tokenizer` protocol from swift-transformers that mlx-swift-lm
/// relies on. Implementations can wrap a Hugging Face tokenizer loaded via
/// a JSON config (vocab, merges, etc.) or any other encoding scheme.
library;

/// Converts between text and token IDs.
abstract interface class Tokenizer {
  /// Encodes [text] into a list of token IDs.
  List<int> encode(String text);

  /// Decodes [tokens] back to a string.
  String decode(List<int> tokens);

  /// Decodes a single [token] ID.
  String decodeToken(int token);

  /// The end-of-sequence token ID, if known.
  int? get eosTokenId;

  /// The unknown-token ID, if known.
  int? get unknownTokenId;

  /// Apply a chat template and return the encoded token IDs.
  ///
  /// [messages] is a list of `{'role': ..., 'content': ...}` maps.
  /// [tools] is an optional list of tool specification objects.
  /// [additionalContext] allows passing model-specific rendering context.
  ///
  /// Throws [MissingChatTemplateException] if no template is available.
  List<int> applyChatTemplate(
    List<Map<String, Object>> messages, {
    List<Object>? tools,
    Map<String, Object>? additionalContext,
  });

  /// Converts a token string to its ID. Returns null if not in vocabulary.
  int? tokenToId(String token);
}

/// Thrown when a tokenizer has no chat template configured.
final class MissingChatTemplateException implements Exception {
  const MissingChatTemplateException([this.message]);
  final String? message;

  @override
  String toString() => 'MissingChatTemplateException: ${message ?? 'No chat template available.'}';
}

/// A minimal tokenizer backed by pre-encoded token IDs.
///
/// Useful for testing or when tokens are pre-processed externally.
final class PassthroughTokenizer implements Tokenizer {
  const PassthroughTokenizer({
    this.eosTokenId,
    this.unknownTokenId,
  });

  @override
  final int? eosTokenId;

  @override
  final int? unknownTokenId;

  @override
  List<int> encode(String text) => throw UnsupportedError('PassthroughTokenizer cannot encode text.');

  @override
  String decode(List<int> tokens) => tokens.join(' ');

  @override
  String decodeToken(int token) => token.toString();

  @override
  List<int> applyChatTemplate(
    List<Map<String, Object>> messages, {
    List<Object>? tools,
    Map<String, Object>? additionalContext,
  }) =>
      throw const MissingChatTemplateException();

  @override
  int? tokenToId(String token) => null;
}

/// Accumulates tokens and provides streaming decoding with proper handling of
/// multi-token UTF-8 sequences (e.g. emoji and non-ASCII characters).
///
/// Mirrors `StreamingDetokenizer` from mlx-swift-lm.
final class StreamingDetokenizer {
  StreamingDetokenizer(this.tokenizer);

  final Tokenizer tokenizer;
  final List<int> _tokens = [];
  String _decoded = '';

  /// Appends [token] and returns any newly decoded text, or null if incomplete.
  String? append(int token) {
    _tokens.add(token);
    final full = tokenizer.decode(_tokens);
    if (full.length > _decoded.length) {
      final delta = full.substring(_decoded.length);
      _decoded = full;
      return delta;
    }
    return null;
  }

  void reset() {
    _tokens.clear();
    _decoded = '';
  }

  String get decoded => _decoded;
}
