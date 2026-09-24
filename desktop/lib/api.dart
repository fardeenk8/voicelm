/// The only file allowed to know the server URL and HTTP verbs.
///
/// Widgets call this class. They do not import `package:http`. That is the UI
/// equivalent of the CLI importing KnowledgeBase and never SQLite.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

class VoiceLmApi {
  VoiceLmApi({http.Client? client, String? baseUrl})
    : _client = client ?? http.Client(),
      baseUrl =
          baseUrl ??
          const String.fromEnvironment(
            'VOICELM_API',
            defaultValue: 'http://127.0.0.1:8000',
          );

  final http.Client _client;
  final String baseUrl;

  Future<List<LibrarySource>> listSources() async {
    final body = await _get('/sources');
    return (body as List<dynamic>)
        .map((item) => LibrarySource.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<LibrarySource> upload({
    required String filename,
    required Uint8List bytes,
  }) async {
    try {
      final request = http.MultipartRequest('POST', _uri('/sources'))
        ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
      final streamed = await _client.send(request);
      final response = await http.Response.fromStream(streamed);
      return LibrarySource.fromJson(_decode(response) as Map<String, dynamic>);
    } on http.ClientException catch (error) {
      throw ApiException(0, _unreachable(error));
    }
  }

  Future<LibrarySource> ingestUrl(String url) async {
    final body = await _post('/sources/url', {'url': url});
    return LibrarySource.fromJson(body as Map<String, dynamic>);
  }

  Future<void> remove(String id) async {
    final response = await _client.delete(_uri('/sources/$id'));
    if (response.statusCode == 204) return;
    throw ApiException(response.statusCode, _detail(response));
  }

  Future<GroundedAnswer> ask(String question, {int topK = 10}) async {
    final body = await _post('/ask', {'question': question, 'top_k': topK});
    return GroundedAnswer.fromJson(body as Map<String, dynamic>);
  }

  /// Tokens as they arrive, then one `AskDoneEvent` with citations.
  ///
  /// `POST /ask` is still available for clients that want one JSON blob.
  /// The desktop screen uses this stream so the answer is not a spinner.
  Stream<AskStreamEvent> askStream(String question, {int topK = 10}) async* {
    try {
      final request = http.Request('POST', _uri('/ask/stream'))
        ..headers['Content-Type'] = 'application/json'
        ..headers['Accept'] = 'text/event-stream'
        ..body = jsonEncode({'question': question, 'top_k': topK});
      final streamed = await _client.send(request);
      if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
        final response = await http.Response.fromStream(streamed);
        throw ApiException(response.statusCode, _detail(response));
      }

      final parser = _SseParser();
      await for (final chunk in streamed.stream.transform(utf8.decoder)) {
        for (final event in parser.add(chunk)) {
          yield _askEvent(event);
        }
      }
      for (final event in parser.flush()) {
        yield _askEvent(event);
      }
    } on http.ClientException catch (error) {
      throw ApiException(0, _unreachable(error));
    }
  }

  AskStreamEvent _askEvent(_SseEvent event) {
    final data = jsonDecode(event.data);
    if (data is! Map<String, dynamic>) {
      throw const ApiException(0, 'unexpected SSE frame from /ask/stream');
    }
    if (event.name == 'token') {
      return AskTokenEvent(data['text'] as String);
    }
    if (event.name == 'done') {
      return AskDoneEvent(GroundedAnswer.fromJson(data));
    }
    if (event.name == 'error') {
      throw ApiException(0, data['detail'] as String? ?? 'generation failed');
    }
    throw ApiException(0, 'unknown SSE event: ${event.name}');
  }

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<dynamic> _get(String path) async {
    try {
      final response = await _client.get(_uri(path));
      return _decode(response);
    } on http.ClientException catch (error) {
      throw ApiException(0, _unreachable(error));
    }
  }

  Future<dynamic> _post(String path, Map<String, dynamic> payload) async {
    try {
      final response = await _client.post(
        _uri(path),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      return _decode(response);
    } on http.ClientException catch (error) {
      throw ApiException(0, _unreachable(error));
    }
  }

  dynamic _decode(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) return null;
      return jsonDecode(response.body);
    }
    throw ApiException(response.statusCode, _detail(response));
  }

  String _detail(http.Response response) {
    try {
      final parsed = jsonDecode(response.body);
      if (parsed is Map<String, dynamic> && parsed['detail'] is String) {
        return parsed['detail'] as String;
      }
    } on FormatException {
      // Fall through to the status line.
    }
    return 'HTTP ${response.statusCode}';
  }

  String _unreachable(http.ClientException error) {
    return 'Cannot reach VoiceLM at $baseUrl. Is the backend running?\n'
        'From backend/: uv run uvicorn --app-dir src voicelm.api.app:app --reload\n'
        '(${error.message})';
  }
}

class _SseEvent {
  const _SseEvent(this.name, this.data);
  final String name;
  final String data;
}

/// Accumulates UTF-8 chunks into `event:` / `data:` frames split by a blank line.
class _SseParser {
  final StringBuffer _buffer = StringBuffer();

  Iterable<_SseEvent> add(String chunk) sync* {
    _buffer.write(chunk);
    yield* _drain(flushRemainder: false);
  }

  Iterable<_SseEvent> flush() => _drain(flushRemainder: true);

  Iterable<_SseEvent> _drain({required bool flushRemainder}) sync* {
    var text = _buffer.toString().replaceAll('\r\n', '\n');
    while (true) {
      final index = text.indexOf('\n\n');
      if (index < 0) {
        if (flushRemainder && text.trim().isNotEmpty) {
          final event = _parseFrame(text);
          if (event != null) yield event;
          text = '';
        }
        break;
      }
      final event = _parseFrame(text.substring(0, index));
      text = text.substring(index + 2);
      if (event != null) yield event;
    }
    _buffer
      ..clear()
      ..write(text);
  }

  _SseEvent? _parseFrame(String raw) {
    var name = 'message';
    final dataLines = <String>[];
    for (final line in raw.split('\n')) {
      if (line.startsWith(':') || line.isEmpty) continue;
      if (line.startsWith('event:')) {
        name = line.substring(6).trim();
      } else if (line.startsWith('data:')) {
        final value = line.substring(5);
        dataLines.add(value.startsWith(' ') ? value.substring(1) : value);
      }
    }
    if (dataLines.isEmpty) return null;
    return _SseEvent(name, dataLines.join('\n'));
  }
}
