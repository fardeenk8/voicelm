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
    this.locationKind,
    this.originUrl,
  });

  factory LibrarySource.fromJson(Map<String, dynamic> json) {
    return LibrarySource(
      id: json['id'] as String,
      title: json['title'] as String,
      path: json['path'] as String,
      chunkCount: json['chunk_count'] as int,
      pageCount: json['page_count'] as int,
      status: json['status'] as String?,
      locationKind: json['location_kind'] as String?,
      originUrl: json['origin_url'] as String?,
    );
  }

  final String id;
  final String title;
  final String path;
  final int chunkCount;
  final int pageCount;
  final String? status;
  final String? locationKind;
  final String? originUrl;

  String get chunkLabel => chunkCount == 1 ? '1 chunk' : '$chunkCount chunks';

  String get pageLabel {
    if (pageCount == 0) return '';
    if (locationKind == 'timestamp') {
      return pageCount == 1 ? '1 cue' : '$pageCount cues';
    }
    if (locationKind == 'line') {
      return pageCount == 1 ? '1 line' : '$pageCount lines';
    }
    final unit = locationKind ?? 'page';
    return pageCount == 1 ? '1 $unit' : '$pageCount ${unit}s';
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
    this.locationKind,
    this.originUrl,
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
      locationKind: json['location_kind'] as String?,
      originUrl: json['origin_url'] as String?,
    );
  }

  final int marker;
  final String sourceId;
  final String sourceTitle;
  final int startChar;
  final int endChar;
  final List<int> pages;
  final String quote;
  final String? locationKind;
  final String? originUrl;

  String get location {
    final chars = 'characters $startChar–$endChar';
    final parts = <String>[];
    if (originUrl != null && originUrl!.isNotEmpty) {
      parts.add(originUrl!);
    }
    if (pages.isNotEmpty) {
      if (locationKind == 'timestamp') {
        final stamps = pages.map(_formatTimestamp).toList();
        parts.add(stamps.length == 1 ? stamps.first : '${stamps.first}–${stamps.last}');
      } else if (locationKind == 'line') {
        parts.add(pages.length == 1 ? 'line ${pages.first}' : 'lines ${pages.first}–${pages.last}');
      } else {
        final unit = locationKind ?? 'page';
        parts.add(pages.length == 1 ? '$unit ${pages.first}' : '${unit}s ${pages.first}–${pages.last}');
      }
    }
    parts.add(chars);
    return parts.join(', ');
  }

  static String _formatTimestamp(int seconds) {
    final safe = seconds < 0 ? 0 : seconds;
    final hours = safe ~/ 3600;
    final minutes = (safe % 3600) ~/ 60;
    final secs = safe % 60;
    if (hours > 0) {
      return '$hours:${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
    }
    return '$minutes:${secs.toString().padLeft(2, '0')}';
  }
}

class GroundedAnswer {
  const GroundedAnswer({
    required this.text,
    required this.model,
    required this.citations,
    required this.unsupportedMarkers,
    this.isComplete = true,
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

  /// False while tokens are still arriving. Citations are only trustworthy
  /// after the `done` event, because a marker mid-stream might be stripped.
  final bool isComplete;
}

sealed class AskStreamEvent {}

final class AskTokenEvent extends AskStreamEvent {
  AskTokenEvent(this.text);
  final String text;
}

final class AskDoneEvent extends AskStreamEvent {
  AskDoneEvent(this.answer);
  final GroundedAnswer answer;
}

class ApiException implements Exception {
  const ApiException(this.statusCode, this.message);

  final int statusCode;
  final String message;

  @override
  String toString() => message;
}
