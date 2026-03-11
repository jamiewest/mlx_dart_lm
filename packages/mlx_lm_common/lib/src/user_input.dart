import 'chat.dart';

/// Raw input from the application layer before tokenization.
///
/// A [UserInputProcessor] converts this into an [LMInput] that a
/// [LanguageModel] can consume. Mirrors `UserInput` from mlx-swift-lm.
final class UserInput {
  UserInput({
    required this.prompt,
    this.images = const [],
    this.tools,
    this.additionalContext,
    this.processing = const Processing(),
  });

  /// Convenience constructor for a plain text prompt.
  UserInput.text(
    String text, {
    this.tools,
    this.additionalContext,
    this.processing = const Processing(),
  })  : prompt = Prompt.chat([ChatMessage.user(text)]),
        images = const [];

  /// Convenience constructor for a structured chat conversation.
  UserInput.chat(
    List<ChatMessage> messages, {
    this.images = const [],
    this.tools,
    this.additionalContext,
    this.processing = const Processing(),
  }) : prompt = Prompt.chat(messages);

  final Prompt prompt;
  final List<Object> images;
  final List<Object>? tools;
  final Map<String, Object>? additionalContext;
  final Processing processing;

  @override
  String toString() => prompt.toString();
}

/// The prompt payload of a [UserInput].
sealed class Prompt {
  const Prompt();

  const factory Prompt.text(String text) = _TextPrompt;
  const factory Prompt.messages(List<Map<String, Object>> messages) = _MessagesPrompt;
  const factory Prompt.chat(List<ChatMessage> messages) = _ChatPrompt;

  @override
  String toString() => switch (this) {
        _TextPrompt(text: final t) => t,
        _MessagesPrompt(messages: final m) => m.map((e) => e.toString()).join('\n'),
        _ChatPrompt(messages: final m) => m.map((e) => e.content).join('\n'),
      };
}

final class _TextPrompt extends Prompt {
  const _TextPrompt(this.text);
  final String text;
}

final class _MessagesPrompt extends Prompt {
  const _MessagesPrompt(this.messages);
  final List<Map<String, Object>> messages;
}

final class _ChatPrompt extends Prompt {
  const _ChatPrompt(this.messages);
  final List<ChatMessage> messages;
}

/// Options for pre-processing media before inference.
final class Processing {
  const Processing({this.resizeWidth, this.resizeHeight});

  final int? resizeWidth;
  final int? resizeHeight;
}
