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

  Future<void> remove(String id) async {
    final response = await _client.delete(_uri('/sources/$id'));
    if (response.statusCode == 204) return;
    throw ApiException(response.statusCode, _detail(response));
  }

  Future<GroundedAnswer> ask(String question, {int topK = 5}) async {
    final body = await _post('/ask', {'question': question, 'top_k': topK});
    return GroundedAnswer.fromJson(body as Map<String, dynamic>);
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
