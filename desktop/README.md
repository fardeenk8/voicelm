# VoiceLM desktop

Flutter macOS client. Presentation only — it talks to the Python backend over HTTP
and never opens SQLite, Qdrant, or Ollama.

```bash
# terminal 1, from backend/
uv run uvicorn --app-dir src voicelm.api.app:app --reload

# terminal 2, from desktop/
flutter run -d macos
```

`lib/api.dart` is the only file that knows the server URL.
