import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:uuid/uuid.dart';

import 'api.dart';
import 'chat_models.dart';
import 'chat_store.dart';
import 'models.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.api, this.store});

  final VoiceLmApi api;
  final ChatStore? store;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const _uuid = Uuid();

  late final ChatStore _store = widget.store ?? ChatStore();
  final _composer = TextEditingController();
  final _scroll = ScrollController();

  ChatSession? _active;
  List<LibrarySource> _sources = [];
  String? _error;
  bool _busy = false;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _composer.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _bootstrap() async {
    try {
      await _store.load();
      try {
        _sources = await widget.api.listSources();
      } on ApiException catch (error) {
        _error = error.message;
      }
      if (_store.sessions.isEmpty) {
        _active = await _store.create();
      } else {
        _active = _store.sessions.first;
      }
    } catch (error) {
      _error = error.toString();
      try {
        _active = await _store.create();
      } catch (_) {
        // Keep UI usable even if persistence is unavailable.
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _newChat() async {
    final session = await _store.create();
    setState(() {
      _active = session;
      _error = null;
      _composer.clear();
    });
  }

  Future<void> _selectChat(ChatSession session) async {
    setState(() {
      _active = session;
      _error = null;
    });
  }

  Future<void> _deleteChat(ChatSession session) async {
    await _store.delete(session.id);
    if (_active?.id == session.id) {
      if (_store.sessions.isEmpty) {
        _active = await _store.create();
      } else {
        _active = _store.sessions.first;
      }
    }
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _refreshLibrary() async {
    try {
      final sources = await widget.api.listSources();
      if (!mounted) return;
      setState(() {
        _sources = sources;
        _error = null;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    }
  }

  static const _documentExtensions = [
    'txt',
    'md',
    'markdown',
    'pdf',
    'docx',
    'pptx',
  ];

  static const _imageExtensions = [
    'png',
    'jpg',
    'jpeg',
    'webp',
    'tif',
    'tiff',
  ];

  static const _mediaExtensions = [
    'mp3',
    'wav',
    'm4a',
    'ogg',
    'flac',
    'mp4',
    'mov',
    'webm',
    'mkv',
  ];

  Future<void> _attachFiles({required List<String> extensions}) async {
    if (_busy) return;
    try {
      final files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: extensions,
      );
      if (files.isEmpty) return;
      final file = files.single;
      final bytes = await file.readAsBytes();
      setState(() {
        _busy = true;
        _error = null;
      });
      final result = await widget.api.upload(filename: file.name, bytes: bytes);
      await _refreshLibrary();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('${result.title}: ${result.status ?? 'added to library'}')),
      );
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } on PlatformException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message ?? error.code);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _attachLink() async {
    if (_busy) return;
    final controller = TextEditingController();
    final url = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Add a link'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'https://…, YouTube, or GitHub repo',
            border: OutlineInputBorder(),
          ),
          onSubmitted: (value) => Navigator.pop(context, value.trim()),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Add'),
          ),
        ],
      ),
    );
    if (url == null || url.isEmpty) return;

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final result = await widget.api.ingestUrl(url);
      await _refreshLibrary();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('${result.title}: ${result.status ?? 'added to library'}')),
      );
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _onAttach(_AttachKind kind) {
    switch (kind) {
      case _AttachKind.document:
        _attachFiles(extensions: _documentExtensions);
      case _AttachKind.image:
        _attachFiles(extensions: _imageExtensions);
      case _AttachKind.media:
        _attachFiles(extensions: _mediaExtensions);
      case _AttachKind.link:
        _attachLink();
    }
  }

  Future<void> _ask() async {
    final question = _composer.text.trim();
    final session = _active;
    if (question.isEmpty || session == null || _busy) return;
    if (_sources.isEmpty) {
      setState(() => _error = 'Add documents in Library before asking.');
      return;
    }

    final user = ChatMessage(
      id: _uuid.v4(),
      role: ChatRole.user,
      text: question,
      createdAt: DateTime.now(),
    );
    final assistantId = _uuid.v4();
    var assistant = ChatMessage(
      id: assistantId,
      role: ChatRole.assistant,
      text: '',
      createdAt: DateTime.now(),
      isComplete: false,
    );

    _composer.clear();
    await _store.saveMessage(session, user);
    await _store.saveMessage(session, assistant);
    setState(() {
      _busy = true;
      _error = null;
    });
    _scrollToBottom();

    try {
      var draft = '';
      await for (final event in widget.api.askStream(question)) {
        if (!mounted) return;
        if (event is AskTokenEvent) {
          draft += event.text;
          assistant = assistant.copyWith(text: draft, isComplete: false);
          await _store.saveMessage(session, assistant);
          setState(() {});
          _scrollToBottom();
        } else if (event is AskDoneEvent) {
          assistant = assistant.copyWith(
            text: event.answer.text,
            citations: event.answer.citations,
            model: event.answer.model,
            isComplete: true,
          );
          await _store.saveMessage(session, assistant);
          setState(() {});
          _scrollToBottom();
        }
      }
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent + 80,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _openLibrary() async {
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      builder: (context) {
        return SizedBox(
          height: MediaQuery.sizeOf(context).height * 0.85,
          child: _LibraryPanel(
            api: widget.api,
            sources: _sources,
            onChanged: (sources) {
              setState(() => _sources = sources);
            },
          ),
        );
      },
    );
    await _refreshLibrary();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      body: Row(
        children: [
          SizedBox(width: 268, child: _sidebar(theme)),
          VerticalDivider(width: 1, color: theme.colorScheme.outlineVariant.withValues(alpha: 0.5)),
          Expanded(child: _chatPane(theme)),
        ],
      ),
    );
  }

  Widget _sidebar(ThemeData theme) {
    final sessions = _store.sessions;
    return ColoredBox(
      color: const Color(0xFFF4F4F5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 18, 14, 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'VoiceLM',
                  style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  onPressed: _busy ? null : _newChat,
                  icon: const Icon(Icons.edit_square, size: 18),
                  label: const Text('New chat'),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 8, 18, 4),
            child: Text(
              'Chats',
              style: theme.textTheme.labelMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: sessions.isEmpty
                ? const Center(child: Text('No chats yet'))
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    itemCount: sessions.length,
                    itemBuilder: (context, index) {
                      final session = sessions[index];
                      final selected = session.id == _active?.id;
                      return _ChatTile(
                        session: session,
                        selected: selected,
                        onTap: () => _selectChat(session),
                        onDelete: () => _deleteChat(session),
                      );
                    },
                  ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.fromLTRB(10, 8, 10, 14),
            child: Column(
              children: [
                ListTile(
                  dense: true,
                  leading: const Icon(Icons.menu_book_outlined, size: 20),
                  title: const Text('Library'),
                  subtitle: Text(
                    _sources.isEmpty
                        ? 'No sources yet'
                        : '${_sources.length} source${_sources.length == 1 ? '' : 's'}',
                  ),
                  onTap: _openLibrary,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _chatPane(ThemeData theme) {
    final session = _active;
    final messages = session?.messages ?? [];

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 8),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  session?.title ?? 'VoiceLM',
                  style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (_sources.isEmpty)
                TextButton.icon(
                  onPressed: _openLibrary,
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('Add sources'),
                ),
            ],
          ),
        ),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: _ErrorBanner(message: _error!),
          ),
        Expanded(
          child: messages.isEmpty
              ? _EmptyChat(onOpenLibrary: _sources.isEmpty ? _openLibrary : null)
              : ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    return _MessageBubble(message: messages[index]);
                  },
                ),
        ),
        _Composer(
          controller: _composer,
          busy: _busy,
          enabled: !_busy,
          onSend: _ask,
          onAttach: _onAttach,
        ),
      ],
    );
  }
}

enum _AttachKind { document, image, media, link }

class _ChatTile extends StatelessWidget {
  const _ChatTile({
    required this.session,
    required this.selected,
    required this.onTap,
    required this.onDelete,
  });

  final ChatSession session;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        color: selected ? Colors.white : Colors.transparent,
        borderRadius: BorderRadius.circular(10),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(10),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 4, 10),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    session.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: selected ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                ),
                IconButton(
                  tooltip: 'Delete chat',
                  visualDensity: VisualDensity.compact,
                  iconSize: 16,
                  onPressed: onDelete,
                  icon: Icon(Icons.close, color: theme.colorScheme.outline),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _EmptyChat extends StatelessWidget {
  const _EmptyChat({this.onOpenLibrary});

  final VoidCallback? onOpenLibrary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.auto_awesome, size: 40, color: theme.colorScheme.primary),
              const SizedBox(height: 16),
              Text(
                'Ask anything about your library',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Text(
                'Answers stay grounded in your documents, with citations back to the source.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              if (onOpenLibrary != null) ...[
                const SizedBox(height: 20),
                FilledButton(onPressed: onOpenLibrary, child: const Text('Open library')),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  const _MessageBubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isUser = message.role == ChatRole.user;

    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Align(
        alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
        child: ConstrainedBox(
          constraints: BoxConstraints(
            maxWidth: MediaQuery.sizeOf(context).width * 0.62,
          ),
          child: Column(
            crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              DecoratedBox(
                decoration: BoxDecoration(
                  color: isUser ? const Color(0xFFE8EEF0) : theme.colorScheme.surface,
                  borderRadius: BorderRadius.circular(18),
                  border: isUser
                      ? null
                      : Border.all(color: theme.colorScheme.outlineVariant.withValues(alpha: 0.5)),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  child: SelectableText(
                    message.text.isEmpty && !message.isComplete ? '…' : message.text,
                    style: theme.textTheme.bodyLarge,
                  ),
                ),
              ),
              if (!isUser && !message.isComplete)
                Padding(
                  padding: const EdgeInsets.only(top: 8, left: 4),
                  child: Text(
                    'Writing…',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              if (!isUser && message.isComplete && message.citations.isNotEmpty) ...[
                const SizedBox(height: 10),
                Text('Sources', style: theme.textTheme.titleSmall),
                const SizedBox(height: 6),
                for (final citation in message.citations) _CitationCard(citation: citation),
                if (message.model.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4, left: 2),
                    child: Text(
                      message.model,
                      style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.outline),
                    ),
                  ),
              ],
            ],
          ),
        ),
      ),
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
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.45),
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
              preview.length > 220 ? '${preview.substring(0, 220)}…' : preview,
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.busy,
    required this.enabled,
    required this.onSend,
    required this.onAttach,
  });

  final TextEditingController controller;
  final bool busy;
  final bool enabled;
  final VoidCallback onSend;
  final ValueChanged<_AttachKind> onAttach;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 8, 24, 20),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: theme.colorScheme.outlineVariant),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.04),
                blurRadius: 16,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(4, 6, 8, 6),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                PopupMenuButton<_AttachKind>(
                  tooltip: 'Attach',
                  enabled: enabled && !busy,
                  offset: const Offset(0, -8),
                  position: PopupMenuPosition.over,
                  onSelected: onAttach,
                  itemBuilder: (context) => const [
                    PopupMenuItem(
                      value: _AttachKind.document,
                      child: ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.description_outlined, size: 22),
                        title: Text('Upload file'),
                        subtitle: Text('PDF, Markdown, Word, slides'),
                      ),
                    ),
                    PopupMenuItem(
                      value: _AttachKind.image,
                      child: ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.image_outlined, size: 22),
                        title: Text('Upload image'),
                        subtitle: Text('PNG, JPG, WebP, TIFF'),
                      ),
                    ),
                    PopupMenuItem(
                      value: _AttachKind.media,
                      child: ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.perm_media_outlined, size: 22),
                        title: Text('Upload audio or video'),
                        subtitle: Text('Transcribed into the library'),
                      ),
                    ),
                    PopupMenuItem(
                      value: _AttachKind.link,
                      child: ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.link, size: 22),
                        title: Text('Add link'),
                        subtitle: Text('Web, YouTube, or GitHub'),
                      ),
                    ),
                  ],
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child: Icon(
                      Icons.add_circle_outline,
                      size: 28,
                      color: enabled && !busy
                          ? theme.colorScheme.onSurfaceVariant
                          : theme.disabledColor,
                    ),
                  ),
                ),
                Expanded(
                  child: TextField(
                    controller: controller,
                    enabled: enabled,
                    minLines: 1,
                    maxLines: 6,
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => onSend(),
                    decoration: const InputDecoration(
                      hintText: 'Message VoiceLM…',
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 4),
                if (busy)
                  const Padding(
                    padding: EdgeInsets.all(12),
                    child: SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                else
                  IconButton.filled(
                    onPressed: enabled ? onSend : null,
                    icon: const Icon(Icons.arrow_upward, size: 20),
                    tooltip: 'Send',
                  ),
              ],
            ),
          ),
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
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: theme.colorScheme.errorContainer,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Text(message, style: TextStyle(color: theme.colorScheme.onErrorContainer)),
        ),
      ),
    );
  }
}

class _LibraryPanel extends StatefulWidget {
  const _LibraryPanel({
    required this.api,
    required this.sources,
    required this.onChanged,
  });

  final VoiceLmApi api;
  final List<LibrarySource> sources;
  final ValueChanged<List<LibrarySource>> onChanged;

  @override
  State<_LibraryPanel> createState() => _LibraryPanelState();
}

class _LibraryPanelState extends State<_LibraryPanel> {
  late List<LibrarySource> _sources = List.of(widget.sources);
  String? _status;
  String? _error;
  bool _busy = false;

  Future<void> _reload() async {
    final sources = await widget.api.listSources();
    if (!mounted) return;
    setState(() => _sources = sources);
    widget.onChanged(sources);
  }

  Future<void> _addUrl() async {
    final controller = TextEditingController();
    final url = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Add a link'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'https://…, YouTube, or GitHub repo',
            border: OutlineInputBorder(),
          ),
          onSubmitted: (value) => Navigator.pop(context, value.trim()),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Add'),
          ),
        ],
      ),
    );
    if (url == null || url.isEmpty) return;

    setState(() {
      _busy = true;
      _error = null;
      _status = 'Fetching $url…';
    });
    try {
      final result = await widget.api.ingestUrl(url);
      await _reload();
      if (!mounted) return;
      setState(() {
        _status = '${result.title}: ${result.status ?? 'ingested'}';
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

  Future<void> _upload() async {
    try {
      final files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: const [
          'txt',
          'md',
          'markdown',
          'pdf',
          'docx',
          'pptx',
          'png',
          'jpg',
          'jpeg',
          'webp',
          'tif',
          'tiff',
          'mp3',
          'wav',
          'm4a',
          'ogg',
          'flac',
          'mp4',
          'mov',
          'webm',
          'mkv',
        ],
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
      await _reload();
      if (!mounted) return;
      setState(() {
        _status = '${result.title}: ${result.status ?? 'ingested'}';
        _busy = false;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message;
        _status = null;
        _busy = false;
      });
    } on PlatformException catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error.message ?? error.code;
        _status = null;
        _busy = false;
      });
    }
  }

  Future<void> _remove(LibrarySource source) async {
    setState(() => _busy = true);
    try {
      await widget.api.remove(source.id);
      await _reload();
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 8),
          child: Row(
            children: [
              Expanded(
                child: Text('Library', style: theme.textTheme.titleLarge),
              ),
              FilledButton.tonalIcon(
                onPressed: _busy ? null : _upload,
                icon: const Icon(Icons.upload_file, size: 18),
                label: const Text('Add file'),
              ),
              const SizedBox(width: 8),
              FilledButton.tonalIcon(
                onPressed: _busy ? null : _addUrl,
                icon: const Icon(Icons.link, size: 18),
                label: const Text('Add link'),
              ),
            ],
          ),
        ),
        if (_busy) const LinearProgressIndicator(minHeight: 2),
        if (_status != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Text(_status!, style: theme.textTheme.bodySmall),
          ),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
            child: _ErrorBanner(message: _error!),
          ),
        Expanded(
          child: _sources.isEmpty
              ? const Center(
                  child: Text('No sources yet.\nAdd a file or paste a web, YouTube, or GitHub link.'),
                )
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(12, 8, 12, 24),
                  itemCount: _sources.length,
                  itemBuilder: (context, index) {
                    final source = _sources[index];
                    return ListTile(
                      title: Text(source.title),
                      subtitle: Text(
                        [
                          source.chunkLabel,
                          if (source.pageLabel.isNotEmpty) source.pageLabel,
                          if (source.originUrl != null)
                            source.originUrl!.contains('youtube.com') ||
                                    source.originUrl!.contains('youtu.be')
                                ? 'youtube'
                                : 'web',
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
    );
  }
}
