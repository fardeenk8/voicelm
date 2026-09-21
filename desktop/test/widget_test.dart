import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:voicelm_desktop/api.dart';
import 'package:voicelm_desktop/main.dart';

void main() {
  testWidgets('an empty library explains what to do', (tester) async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        return http.Response('[]', 200);
      }),
    );

    await tester.pumpWidget(VoiceLmApp(api: api));
    await tester.pumpAndSettle();

    expect(find.textContaining('No documents yet'), findsOneWidget);
    expect(tester.widget<FilledButton>(find.byType(FilledButton)).onPressed, isNull);
  });

  testWidgets('asking with a source shows the answer and citation', (tester) async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        if (request.method == 'GET') {
          return http.Response(
            jsonEncode([
              {
                'id': '1',
                'title': 'physics',
                'path': '/tmp/physics.md',
                'chunk_count': 1,
                'page_count': 0,
              },
            ]),
            200,
          );
        }
        expect(request.url.path, '/ask/stream');
        return http.Response(
          'event: token\ndata: {"text":"Light is fast [1]."}\n\n'
          'event: done\ndata: ${jsonEncode({
            'text': 'Light is fast [1].',
            'model': 'fake-chat',
            'citations': [
              {
                'marker': 1,
                'source_id': '1',
                'source_title': 'physics',
                'start_char': 0,
                'end_char': 20,
                'pages': <int>[],
                'quote': 'Light travels quickly.',
              },
            ],
            'unsupported_markers': <int>[],
          })}\n\n',
          200,
          headers: {'content-type': 'text/event-stream'},
        );
      }),
    );

    await tester.pumpWidget(VoiceLmApp(api: api));
    await tester.pumpAndSettle();

    expect(find.text('physics'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'how fast is light?');
    await tester.tap(find.widgetWithText(FilledButton, 'Ask'));
    await tester.pumpAndSettle();

    expect(find.text('Light is fast [1].'), findsOneWidget);
    expect(find.textContaining('[1] physics'), findsOneWidget);
  });

  testWidgets('a down backend surfaces the connection error', (tester) async {
    final api = VoiceLmApi(
      baseUrl: 'http://127.0.0.1:9',
      client: MockClient((request) async {
        throw http.ClientException('Connection refused');
      }),
    );

    await tester.pumpWidget(VoiceLmApp(api: api));
    await tester.pumpAndSettle();

    expect(find.textContaining('Cannot reach VoiceLM'), findsOneWidget);
  });
}
