/// JSON shapes from the VoiceLM HTTP API.
///
/// These are views of the backend's contract, not a second knowledge model.
/// Parsing lives here so widgets never touch raw maps.
library;

class LibrarySource {
  const LibrarySource({
    required this.id,
    required this.title,
    required this.path,
    required this.chunkCount,
    required this.pageCount,
    this.status,
  });

  factory LibrarySource.fromJson(Map<String, dynamic> json) {
    return LibrarySource(
      id: json['id'] as String,
      title: json['title'] as String,
      path: json['path'] as String,
      chunkCount: json['chunk_count'] as int,
      pageCount: json['page_count'] as int,
      status: json['status'] as String?,
    );
  }

  final String id;
  final String title;
  final String path;
  final int chunkCount;
  final int pageCount;
  final String? status;

  String get chunkLabel => chunkCount == 1 ? '1 chunk' : '$chunkCount chunks';

  String get pageLabel {
    if (pageCount == 0) return '';
    return pageCount == 1 ? '1 page' : '$pageCount pages';
  }
}

class AnswerCitation {
  const AnswerCitation({
    required this.marker,
    required this.sourceId,
    required this.sourceTitle,
    required this.startChar,
    required this.endChar,
    required this.pages,
    required this.quote,
  });

  factory AnswerCitation.fromJson(Map<String, dynamic> json) {
    return AnswerCitation(
      marker: json['marker'] as int,
      sourceId: json['source_id'] as String,
      sourceTitle: json['source_title'] as String,
      startChar: json['start_char'] as int,
      endChar: json['end_char'] as int,
      pages: (json['pages'] as List<dynamic>).cast<int>(),
      quote: json['quote'] as String,
    );
  }

  final int marker;
  final String sourceId;
  final String sourceTitle;
  final int startChar;
  final int endChar;
  final List<int> pages;
  final String quote;

  String get location {
    final chars = 'characters $startChar–$endChar';
    if (pages.isEmpty) return chars;
    if (pages.length == 1) return 'page ${pages.first}, $chars';
    return 'pages ${pages.first}–${pages.last}, $chars';
  }
}

class GroundedAnswer {
  const GroundedAnswer({
    required this.text,
    required this.model,
    required this.citations,
    required this.unsupportedMarkers,
  });

  factory GroundedAnswer.fromJson(Map<String, dynamic> json) {
    return GroundedAnswer(
      text: json['text'] as String,
      model: json['model'] as String,
      citations: (json['citations'] as List<dynamic>)
          .map((item) => AnswerCitation.fromJson(item as Map<String, dynamic>))
          .toList(),
      unsupportedMarkers: (json['unsupported_markers'] as List<dynamic>).cast<int>(),
    );
  }

  final String text;
  final String model;
  final List<AnswerCitation> citations;
  final List<int> unsupportedMarkers;
}

class ApiException implements Exception {
  const ApiException(this.statusCode, this.message);

  final int statusCode;
  final String message;

  @override
  String toString() => message;
}
