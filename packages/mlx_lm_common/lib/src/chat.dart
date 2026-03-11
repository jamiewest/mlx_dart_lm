/// The role of a chat participant.
enum ChatRole {
  system,
  user,
  assistant,
  tool,
}

/// A single message in a multi-turn conversation.
final class ChatMessage {
  const ChatMessage({
    required this.role,
    required this.content,
    this.images = const [],
  });

  final ChatRole role;
  final String content;

  /// Any images attached to this message (for VLMs).
  final List<Object> images;

  factory ChatMessage.system(String content) =>
      ChatMessage(role: ChatRole.system, content: content);

  factory ChatMessage.user(String content, {List<Object> images = const []}) =>
      ChatMessage(role: ChatRole.user, content: content, images: images);

  factory ChatMessage.assistant(String content) =>
      ChatMessage(role: ChatRole.assistant, content: content);

  @override
  String toString() => '${role.name}: $content';
}
