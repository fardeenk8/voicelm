/// Persist chat sessions under Application Support as one JSON file.
library;

import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

import 'chat_models.dart';

class ChatStore {
  /// Disk-backed store under Application Support.
  ChatStore() : _rootDir = null, _memory = false;

  /// Disk-backed store under [rootDir] (tests that need real files).
  ChatStore.root(Directory rootDir) : _rootDir = rootDir, _memory = false;

  /// In-memory only — widget tests, no dart:io.
  ChatStore.memory() : _rootDir = null, _memory = true;

  static const _uuid = Uuid();
  final Directory? _rootDir;
  final bool _memory;
  List<ChatSession> _sessions = [];
  bool _loaded = false;

  List<ChatSession> get sessions {
    final copy = List<ChatSession>.from(_sessions);
    copy.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return copy;
  }

  Future<void> load() async {
    if (_loaded) return;
    if (_memory) {
      _loaded = true;
      return;
    }
    final file = await _file();
    if (await file.exists()) {
      try {
        final decoded = jsonDecode(await file.readAsString());
        if (decoded is List) {
          _sessions = decoded
              .map((item) => ChatSession.fromJson(item as Map<String, dynamic>))
              .toList();
        }
      } on FormatException {
        _sessions = [];
      }
    }
    _loaded = true;
  }

  Future<ChatSession> create() async {
    await load();
    final now = DateTime.now();
    final session = ChatSession(
      id: _uuid.v4(),
      title: 'New chat',
      createdAt: now,
      updatedAt: now,
    );
    _sessions.insert(0, session);
    await _save();
    return session;
  }

  ChatSession? get(String id) {
    for (final session in _sessions) {
      if (session.id == id) return session;
    }
    return null;
  }

  Future<void> touch(ChatSession session) async {
    session.updatedAt = DateTime.now();
    await _save();
  }

  Future<void> rename(ChatSession session, String title) async {
    session.title = title.trim().isEmpty ? 'New chat' : title.trim();
    session.updatedAt = DateTime.now();
    await _save();
  }

  Future<void> delete(String id) async {
    _sessions.removeWhere((session) => session.id == id);
    await _save();
  }

  Future<void> saveMessage(ChatSession session, ChatMessage message) async {
    final index = session.messages.indexWhere((item) => item.id == message.id);
    if (index >= 0) {
      session.messages[index] = message;
    } else {
      session.messages.add(message);
    }
    if (session.title == 'New chat' && message.role == ChatRole.user) {
      session.title = _titleFrom(message.text);
    }
    session.updatedAt = DateTime.now();
    await _save();
  }

  String _titleFrom(String text) {
    final cleaned = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (cleaned.length <= 42) return cleaned.isEmpty ? 'New chat' : cleaned;
    return '${cleaned.substring(0, 42).trimRight()}…';
  }

  Future<void> _save() async {
    if (_memory) return;
    final file = await _file();
    await file.parent.create(recursive: true);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_sessions.map((s) => s.toJson()).toList()),
    );
  }

  Future<File> _file() async {
    final dir = _rootDir ?? await getApplicationSupportDirectory();
    return File('${dir.path}/chats.json');
  }
}
