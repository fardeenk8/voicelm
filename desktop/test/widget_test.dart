import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:voicelm_desktop/api.dart';
import 'package:voicelm_desktop/chat_store.dart';
import 'package:voicelm_desktop/main.dart';

void main() {
  Future<void> pumpUntilFound(WidgetTester tester, Finder finder) async {
    for (var i = 0; i < 40; i++) {
      await tester.pump(const Duration(milliseconds: 50));
      if (finder.evaluate().isNotEmpty) return;
    }
    fail('Timed out waiting for $finder');
  }

  testWidgets('empty state invites asking about the library', (tester) async {
    final api = VoiceLmApi(
      baseUrl: 'http://test',
      client: MockClient((request) async {
        return http.Response('[]', 200);
      }),
    );

    await tester.pumpWidget(
      VoiceLmApp(api: api, store: ChatStore.memory()),
    );
    await pumpUntilFound(tester, find.textContaining('Ask anything about your library'));

    expect(find.textContaining('Ask anything about your library'), findsOneWidget);
    expect(find.text('New chat'), findsWidgets);
    expect(find.text('Library'), findsOneWidget);
  });

  testWidgets('asking with a source streams the answer into the chat', (tester) async {
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
            headers: {'content-type': 'application/json'},
          );
        }
        if (request.url.path.endsWith('/ask/stream')) {
          final body =
              'event: token\ndata: {"text":"Light is fast [1]."}\n\n'
              'event: done\ndata: ${jsonEncode({
                'text': 'Light is fast [1].',
                'model': 'fake',
                'citations': [
                  {
                    'marker': 1,
                    'source_id': '1',
                    'source_title': 'physics',
                    'start_char': 0,
                    'end_char': 14,
                    'pages': <int>[],
                    'quote': 'Light is fast.',
                  },
                ],
                'unsupported_markers': <int>[],
              })}\n\n';
          return http.Response(
            body,
            200,
            headers: {'content-type': 'text/event-stream'},
          );
        }
        return http.Response('unexpected', 500);
      }),
    );

    await tester.pumpWidget(
      VoiceLmApp(api: api, store: ChatStore.memory()),
    );
    await pumpUntilFound(tester, find.byType(TextField));

    await tester.enterText(find.byType(TextField), 'How fast is light?');
    await tester.tap(find.byTooltip('Send'));
    await pumpUntilFound(tester, find.textContaining('Light is fast'));

    expect(find.textContaining('Light is fast'), findsWidgets);
    expect(find.textContaining('physics'), findsWidgets);
  });
}
