import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:voicelm_desktop/api.dart';
import 'package:voicelm_desktop/models.dart';

void main() {
  test('listSources decodes the array the API returns', () async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        expect(request.method, 'GET');
        expect(request.url.path, '/sources');
        return http.Response(
          jsonEncode([
            {
              'id': '1',
              'title': 'notes',
              'path': '/tmp/notes.md',
              'chunk_count': 1,
              'page_count': 0,
            },
          ]),
          200,
        );
      }),
    );

    final sources = await api.listSources();

    expect(sources.single.title, 'notes');
  });

  test('ask sends the question and parses citations', () async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        expect(request.method, 'POST');
        expect(jsonDecode(request.body)['question'], 'how fast?');
        return http.Response(
          jsonEncode({
            'text': 'Very [1].',
            'model': 'fake',
            'citations': [
              {
                'marker': 1,
                'source_id': 's',
                'source_title': 'physics',
                'start_char': 0,
                'end_char': 4,
                'pages': <int>[],
                'quote': 'fast',
              },
            ],
            'unsupported_markers': <int>[],
          }),
          200,
        );
      }),
    );

    final answer = await api.ask('how fast?');

    expect(answer.text, 'Very [1].');
    expect(answer.citations.single.sourceTitle, 'physics');
  });

  test('a 409 empty library becomes an ApiException with the detail', () async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        return http.Response(jsonEncode({'detail': 'library is empty'}), 409);
      }),
    );

    expect(
      () => api.ask('anything?'),
      throwsA(isA<ApiException>().having((e) => e.message, 'message', 'library is empty')),
    );
  });

  test('remove treats 204 as success', () async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        expect(request.method, 'DELETE');
        return http.Response('', 204);
      }),
    );

    await api.remove('abc');
  });
}
