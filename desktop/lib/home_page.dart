import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api.dart';
import 'models.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.api});

  final VoiceLmApi api;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _question = TextEditingController();

  List<LibrarySource> _sources = [];
  GroundedAnswer? _answer;
  String? _error;
  String? _status;
  bool _loadingLibrary = true;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refreshLibrary();
  }

  @override
  void dispose() {
    _question.dispose();
    super.dispose();
  }

  Future<void> _refreshLibrary() async {
    setState(() {
      _loadingLibrary = true;
      _error = null;
    });
    try {
      final sources = await widget.api.listSources();
      if (!mounted) return;
      setState(() {
        _sources = sources;
        _loadingLibrary = false;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _loadingLibrary = false;
      });
    }
  }

  Future<void> _upload() async {
    try {
      final files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['txt', 'md', 'markdown', 'pdf'],
      );
      if (files.isEmpty) return;

      final file = files.single;
      final bytes = await file.readAsBytes();

      setState(() {
        _busy = true;
        _error = null;
        _status = 'Ingesting ${file.name}…';
      });
      final result = await widget.api.upload(filename: file.name, bytes: bytes);
      await _refreshLibrary();
      if (!mounted) return;
      setState(() {
        _status = '${result.title}: ${result.status ?? 'ingested'}, ${result.chunkLabel}';
        _busy = false;
      });
    } on ApiException catch (error) {
      _showUploadError(error.message);
    } on PlatformException catch (error) {
      _showUploadError(error.message ?? error.code);
    } catch (error) {
      _showUploadError(error.toString());
    }
  }

  void _showUploadError(String message) {
    if (!mounted) return;
    setState(() {
      _error = message;
      _status = null;
      _busy = false;
    });
  }

  Future<void> _remove(LibrarySource source) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Remove ${source.title}?'),
        content: const Text(
          'This drops it from the library. The original file on disk is not deleted '
          'unless VoiceLM stored its own copy.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('Remove')),
        ],
      ),
    );
    if (confirmed != true) return;

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await widget.api.remove(source.id);
      if (_answer?.citations.any((citation) => citation.sourceId == source.id) ?? false) {
        _answer = null;
      }
      await _refreshLibrary();
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _ask() async {
    final question = _question.text.trim();
    if (question.isEmpty) return;

    setState(() {
      _busy = true;
      _error = null;
      _status = 'Searching…';
      _answer = null;
    });
    try {
      final answer = await widget.api.ask(question);
      if (!mounted) return;
      setState(() {
        _answer = answer;
        _status = null;
        _busy = false;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _status = null;
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          SizedBox(width: 320, child: _libraryPane(context)),
          const VerticalDivider(width: 1),
          Expanded(child: _askPane(context)),
        ],
      ),
    );
  }

  Widget _libraryPane(BuildContext context) {
    final theme = Theme.of(context);
    return ColoredBox(
      color: theme.colorScheme.surfaceContainerLowest,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 12, 8),
            child: Row(
              children: [
                Expanded(child: Text('Library', style: theme.textTheme.titleLarge)),
                IconButton(
                  tooltip: 'Add a document',
                  onPressed: _busy ? null : _upload,
                  icon: const Icon(Icons.add),
                ),
                IconButton(
                  tooltip: 'Refresh',
                  onPressed: _busy ? null : _refreshLibrary,
                  icon: const Icon(Icons.refresh),
                ),
              ],
            ),
          ),
          if (_loadingLibrary) const LinearProgressIndicator(minHeight: 2),
          Expanded(
            child: _sources.isEmpty && !_loadingLibrary
                ? const _EmptyHint(
                    icon: Icons.folder_open,
                    text: 'No documents yet.\nAdd a .txt, .md, or .pdf.',
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: _sources.length,
                    itemBuilder: (context, index) {
                      final source = _sources[index];
                      return ListTile(
                        title: Text(source.title),
                        subtitle: Text(
                          [
                            source.chunkLabel,
                            if (source.pageLabel.isNotEmpty) source.pageLabel,
                          ].join(' · '),
                        ),
                        trailing: IconButton(
                          tooltip: 'Remove',
                          onPressed: _busy ? null : () => _remove(source),
                          icon: const Icon(Icons.close),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _askPane(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(28, 24, 28, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Ask your documents', style: theme.textTheme.titleLarge),
              const SizedBox(height: 4),
              Text(
                'Answers are grounded in the library and cite the passage they used.',
                style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: TextField(
                  controller: _question,
                  minLines: 2,
                  maxLines: 4,
                  enabled: !_busy,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _ask(),
                  decoration: const InputDecoration(
                    hintText: 'What do you want to know?',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              FilledButton(
                onPressed: _busy || _sources.isEmpty ? null : _ask,
                child: const Text('Ask'),
              ),
            ],
          ),
        ),
        if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator(minHeight: 2)),
        if (_status != null)
          Padding(
            padding: const EdgeInsets.fromLTRB(28, 12, 28, 0),
            child: Text(_status!, style: theme.textTheme.bodySmall),
          ),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.fromLTRB(28, 12, 28, 0),
            child: _ErrorBanner(message: _error!),
          ),
        Expanded(
          child: _answer == null
              ? const _EmptyHint(
                  icon: Icons.chat_bubble_outline,
                  text: 'Ask a question about the documents on the left.',
                )
              : _AnswerView(answer: _answer!),
        ),
      ],
    );
  }
}

class _AnswerView extends StatelessWidget {
  const _AnswerView({required this.answer});

  final GroundedAnswer answer;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.fromLTRB(28, 20, 28, 32),
      children: [
        SelectableText(answer.text, style: theme.textTheme.bodyLarge),
        if (answer.unsupportedMarkers.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: Text(
              'The model referenced ${answer.unsupportedMarkers.map((m) => '[$m]').join(', ')}, '
              'which matched no excerpt; those markers were removed.',
              style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
            ),
          ),
        const SizedBox(height: 24),
        Text('Sources', style: theme.textTheme.titleMedium),
        const SizedBox(height: 8),
        if (answer.citations.isEmpty)
          Text(
            'No citations — this answer is not backed by a specific passage.',
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          )
        else
          for (final citation in answer.citations) _CitationCard(citation: citation),
        const SizedBox(height: 16),
        Text(answer.model, style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.outline)),
      ],
    );
  }
}

class _CitationCard extends StatelessWidget {
  const _CitationCard({required this.citation});

  final AnswerCitation citation;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final preview = citation.quote.replaceAll(RegExp(r'\s+'), ' ');
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '[${citation.marker}] ${citation.sourceTitle}, ${citation.location}',
              style: theme.textTheme.titleSmall,
            ),
            const SizedBox(height: 6),
            Text(
              preview.length > 280 ? '${preview.substring(0, 280)}…' : preview,
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _EmptyHint extends StatelessWidget {
  const _EmptyHint({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 36, color: theme.colorScheme.outline),
            const SizedBox(height: 12),
            Text(
              text,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Text(message, style: TextStyle(color: theme.colorScheme.onErrorContainer)),
      ),
    );
  }
}
