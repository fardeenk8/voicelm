/// Local chat sessions — UI memory only.
///
/// Each ask still hits the RAG API independently. Server-side multi-turn memory
/// is Phase 3; this store makes the desktop feel like ChatGPT in the meantime.
library;

import 'models.dart';

enum ChatRole { user, assistant }

class ChatMessage {
  const ChatMessage({
    required this.id,
    required this.role,
    required this.text,
    required this.createdAt,
    this.citations = const [],
    this.model = '',
    this.isComplete = true,
  });

  final String id;
  final ChatRole role;
  final String text;
  final DateTime createdAt;
  final List<AnswerCitation> citations;
  final String model;
  final bool isComplete;

  ChatMessage copyWith({
    String? text,
    List<AnswerCitation>? citations,
    String? model,
    bool? isComplete,
  }) {
    return ChatMessage(
      id: id,
      role: role,
      text: text ?? this.text,
      createdAt: createdAt,
      citations: citations ?? this.citations,
      model: model ?? this.model,
      isComplete: isComplete ?? this.isComplete,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'role': role.name,
    'text': text,
    'created_at': createdAt.toIso8601String(),
    'citations': citations
        .map(
          (c) => {
            'marker': c.marker,
            'source_id': c.sourceId,
            'source_title': c.sourceTitle,
            'start_char': c.startChar,
            'end_char': c.endChar,
            'pages': c.pages,
            'quote': c.quote,
            'location_kind': c.locationKind,
            'origin_url': c.originUrl,
          },
        )
        .toList(),
    'model': model,
    'is_complete': isComplete,
  };

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      id: json['id'] as String,
      role: ChatRole.values.byName(json['role'] as String),
      text: json['text'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      citations: (json['citations'] as List<dynamic>? ?? [])
          .map((item) => AnswerCitation.fromJson(item as Map<String, dynamic>))
          .toList(),
      model: json['model'] as String? ?? '',
      isComplete: json['is_complete'] as bool? ?? true,
    );
  }
}

class ChatSession {
  ChatSession({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    List<ChatMessage>? messages,
  }) : messages = messages ?? [];

  final String id;
  String title;
  final DateTime createdAt;
  DateTime updatedAt;
  final List<ChatMessage> messages;

  bool get isEmpty => messages.isEmpty;

  Map<String, dynamic> toJson() => {
    'id': id,
    'title': title,
    'created_at': createdAt.toIso8601String(),
    'updated_at': updatedAt.toIso8601String(),
    'messages': messages.map((m) => m.toJson()).toList(),
  };

  factory ChatSession.fromJson(Map<String, dynamic> json) {
    return ChatSession(
      id: json['id'] as String,
      title: json['title'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      updatedAt: DateTime.parse(json['updated_at'] as String),
      messages: (json['messages'] as List<dynamic>? ?? [])
          .map((item) => ChatMessage.fromJson(item as Map<String, dynamic>))
          .toList(),
    );
  }
}
