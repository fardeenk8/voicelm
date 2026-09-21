import 'package:flutter_test/flutter_test.dart';
import 'package:voicelm_desktop/models.dart';

void main() {
  test('LibrarySource reads the API field names', () {
    final source = LibrarySource.fromJson({
      'id': 'abc',
      'title': 'notes',
      'path': '/tmp/files/notes.md',
      'chunk_count': 3,
      'page_count': 0,
      'status': 'ingested',
    });

    expect(source.chunkLabel, '3 chunks');
    expect(source.pageLabel, '');
    expect(source.status, 'ingested');
  });

  test('a PDF citation formats the page the viewer would show', () {
    final citation = AnswerCitation.fromJson({
      'marker': 1,
      'source_id': 'src',
      'source_title': 'paper',
      'start_char': 10,
      'end_char': 40,
      'pages': [2],
      'quote': 'CataractNet uses fundus images.',
    });

    expect(citation.location, 'page 2, characters 10–40');
  });

  test('GroundedAnswer parses citations and unsupported markers', () {
    final answer = GroundedAnswer.fromJson({
      'text': 'Light is fast [1].',
      'model': 'llama3.1:8b',
      'citations': [
        {
          'marker': 1,
          'source_id': 'src',
          'source_title': 'physics',
          'start_char': 0,
          'end_char': 12,
          'pages': <int>[],
          'quote': 'Light is fast.',
        },
      ],
      'unsupported_markers': [2],
    });

    expect(answer.citations.single.sourceTitle, 'physics');
    expect(answer.unsupportedMarkers, [2]);
  });
}
