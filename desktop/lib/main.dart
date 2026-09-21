import 'package:flutter/material.dart';

import 'api.dart';
import 'home_page.dart';

void main() {
  runApp(VoiceLmApp(api: VoiceLmApi()));
}

class VoiceLmApp extends StatelessWidget {
  const VoiceLmApp({super.key, required this.api});

  final VoiceLmApi api;

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF1F4E5F));
    return MaterialApp(
      title: 'VoiceLM',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorScheme: scheme, useMaterial3: true),
      home: HomePage(api: api),
    );
  }
}
