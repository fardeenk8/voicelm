import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'api.dart';
import 'home_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Debug builds are unsigned (File Provider breaks codesign under Documents),
  // so the user-selected-files entitlement never attaches. file_picker then
  // refuses to open a dialog. Skip that check only in debug; signed Release
  // still uses the entitlements in DebugProfile/Release.entitlements.
  if (kDebugMode) {
    await FilePicker.skipEntitlementsChecks();
  }
  runApp(VoiceLmApp(api: VoiceLmApi()));
}

class VoiceLmApp extends StatelessWidget {
  const VoiceLmApp({super.key, required this.api});

  final VoiceLmApi api;

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF1F4E5F));
    final base = ThemeData(colorScheme: scheme, useMaterial3: true);
    return MaterialApp(
      title: 'VoiceLM',
      debugShowCheckedModeBanner: false,
      theme: base.copyWith(
        textTheme: _readable(base.textTheme),
        primaryTextTheme: _readable(base.primaryTextTheme),
        inputDecorationTheme: const InputDecorationTheme(
          hintStyle: TextStyle(fontFamily: 'SourceSans3', wordSpacing: 4),
        ),
      ),
      home: HomePage(api: api),
    );
  }
}

/// Bundled font plus extra word spacing.
///
/// On this Mac, system fonts (and Helvetica Neue) were drawn with a zero-width
/// space glyph, so "Ask your documents" appeared as "Askyourdocuments".
/// `wordSpacing` is extra width at each whitespace run — it still applies if
/// the space character itself has no advance.
TextTheme _readable(TextTheme theme) {
  TextStyle? fix(TextStyle? style) {
    return style?.copyWith(fontFamily: 'SourceSans3', wordSpacing: 4);
  }

  return theme.copyWith(
    displayLarge: fix(theme.displayLarge),
    displayMedium: fix(theme.displayMedium),
    displaySmall: fix(theme.displaySmall),
    headlineLarge: fix(theme.headlineLarge),
    headlineMedium: fix(theme.headlineMedium),
    headlineSmall: fix(theme.headlineSmall),
    titleLarge: fix(theme.titleLarge),
    titleMedium: fix(theme.titleMedium),
    titleSmall: fix(theme.titleSmall),
    bodyLarge: fix(theme.bodyLarge),
    bodyMedium: fix(theme.bodyMedium),
    bodySmall: fix(theme.bodySmall),
    labelLarge: fix(theme.labelLarge),
    labelMedium: fix(theme.labelMedium),
    labelSmall: fix(theme.labelSmall),
  );
}
