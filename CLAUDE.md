# Veritas Reader — Project Guide for Claude

## Project Overview

**Veritas Reader** is a local-first desktop application built with Python that combines document import/editing, local LLM text generation via Ollama, and high-fidelity text-to-speech (TTS) synthesis. The goal is a polished, all-in-one tool: import text → optionally generate/refine with AI → edit → synthesize to audio → export.

---

## Tech Stack

| Concern              | Choice                                              |
|----------------------|-----------------------------------------------------|
| GUI Framework        | **PyQt6**                                           |
| Document Parsing     | `python-docx`, `markdown2`, `markdownify`           |
| Google Docs          | `google-api-python-client`, `google-auth-oauthlib`  |
| Local LLM            | **Ollama** (HTTP API at `http://localhost:11434`)   |
| TTS Engine           | **Kokoro-82M** (via `kokoro` Python package) — priority: quality |
| TTS Fallback         | **Coqui TTS** (`TTS` package)                       |
| Audio Post-Processing| `pydub`, `ffmpeg-python`                            |
| HTTP Client          | `httpx` (async), `requests` (sync fallback)         |
| Markdown Editor      | `QTextEdit` with custom markdown renderer           |
| Rich Text Export     | `python-docx`, `markdown2`                          |

---

## Project Structure

```
veritas_reader/
├── CLAUDE.md                    # This file
├── requirements.txt
├── main.py                      # Entry point — launches PyQt6 app
├── app/
│   ├── __init__.py
│   ├── window.py                # MainWindow (QMainWindow subclass)
│   ├── toolbar.py               # ModelSelector, VoiceSelector, PlaybackControls, FileNaming
│   ├── editor.py                # Rich-text / markdown editor widget
│   ├── player.py                # Built-in audio player widget (QMediaPlayer)
│   └── dialogs/
│       ├── paste_dialog.py      # Paste Plain Text dialog
│       └── gdocs_dialog.py      # Google Docs OAuth dialog
├── core/
│   ├── __init__.py
│   ├── file_handler.py          # Open/parse .md, .txt, .docx files
│   ├── gdocs.py                 # Google Docs API integration
│   ├── ollama_client.py         # Ollama HTTP client (list models, generate)
│   ├── tts_engine.py            # TTS abstraction (Kokoro primary, Coqui fallback)
│   └── audio_processor.py      # Silence removal, wav/mp3 export via pydub/ffmpeg
├── assets/
│   └── icons/                   # UI icons
├── config/
│   ├── settings.py              # App-wide settings (QSettings backed)
│   └── google_oauth.json        # Google OAuth credentials (gitignored)
└── tests/
    ├── test_ollama_client.py
    ├── test_tts_engine.py
    └── test_file_handler.py
```

---

## Core Module Contracts

### `core/ollama_client.py`
- `list_models() -> list[str]` — calls `GET /api/tags`, returns model name list
- `generate(model: str, prompt: str, stream: bool = True) -> Iterator[str]` — streams tokens
- `chat(model: str, messages: list[dict], stream: bool = True) -> Iterator[str]` — chat completion
- Ollama base URL: `http://localhost:11434` (configurable via `OLLAMA_HOST` env var)

### `core/tts_engine.py`
- `TTSEngine` abstract base class with `.synthesize(text: str, output_path: Path) -> Path`
- `KokoroEngine(TTSEngine)` — primary, uses `kokoro` package
- `CoquiEngine(TTSEngine)` — fallback, uses `TTS` package
- Voice profiles stored as dicts: `{"id": str, "name": str, "engine": str}`
- `list_voices() -> list[VoiceProfile]`

### `core/audio_processor.py`
- `remove_silence(input_path: Path, output_path: Path, silence_thresh_db: int = -40, min_silence_ms: int = 300) -> Path`
- `export_wav(audio: AudioSegment, path: Path) -> Path`
- `export_mp3(audio: AudioSegment, path: Path, bitrate: str = "192k") -> Path`
- Uses `pydub.silence.split_on_silence` then concatenation

### `core/file_handler.py`
- `read_file(path: Path) -> str` — dispatches by extension (.md, .txt, .docx)
- `write_docx(text: str, path: Path) -> None`
- `write_markdown(text: str, path: Path) -> None`
- Markdown parsed with `markdown2`; .docx with `python-docx`

### `app/toolbar.py`
- `ModelSelectorCombo(QComboBox)` — populated async on app start, refresh button
- `VoiceSelectorCombo(QComboBox)` — lists TTS voice profiles
- `PlaybackControls(QWidget)` — play/pause/stop/seek using `QMediaPlayer`
- `FileNameInput(QLineEdit)` — output filename (no extension, added on export)

---

## Key UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [Open File] [Paste Text] [Google Docs]  │ File: [________] │
├─────────────────────────────────────────────────────────────┤
│ Model: [dropdown▼]  Voice: [dropdown▼]  [Generate] [TTS]   │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│       EDITOR (left)          │    AI PROMPT / OUTPUT        │
│   (markdown/rich-text)       │    (QTextEdit)               │
│                              │                              │
├──────────────────────────────┴──────────────────────────────┤
│  [▶ Play] [⏸] [⏹]  ████████░░░░  00:00 / 00:00            │
├─────────────────────────────────────────────────────────────┤
│ [Export .md] [Export .docx] [Sync Google Docs] [Export WAV] │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Conventions

- **Python 3.11+** required.
- All long-running operations (Ollama generation, TTS synthesis, file I/O) run in `QThread` workers — never block the main thread.
- Worker pattern: `class WorkerSignals(QObject)` + `class Worker(QRunnable)` using `QThreadPool`.
- Use `QSettings("VeritasReader", "VeritasReader")` for persisting user preferences (last model, voice, output dir).
- Error handling: show `QMessageBox.critical()` for user-facing errors; log to `logging` module (DEBUG level by default in dev).
- No internet access beyond Google Docs OAuth and Ollama's local HTTP server.
- All audio files generated to a temp dir first, then moved to user-selected output on export.
- **Do not hard-code paths.** Use `pathlib.Path` and `platformdirs` for config/data dirs.

---

## Audio Post-Processing Rules

1. After TTS synthesis, always run `remove_silence()` before playback or export.
2. Silence threshold: `-40 dBFS`, minimum silence length: `300ms` (tunable in settings).
3. Add `50ms` of silence padding between rejoined chunks for naturalness.
4. Default export format: `.wav` (lossless). Optional `.mp3` at 192k.
5. Never overwrite the raw TTS output — keep it as `_raw.wav` alongside the processed version.

---

## Google Docs Integration

- OAuth2 flow using `google-auth-oauthlib`.
- Scopes: `https://www.googleapis.com/auth/documents` and `https://www.googleapis.com/auth/drive.file`.
- Credentials stored in `~/.config/veritas_reader/token.json` (not in project dir).
- `gdocs.py` exposes: `authenticate()`, `read_doc(doc_id) -> str`, `write_doc(doc_id, text) -> None`, `create_doc(title, text) -> str`.

---

## requirements.txt (Canonical)

```
PyQt6>=6.7.0
PyQt6-Qt6>=6.7.0
httpx>=0.27.0
requests>=2.32.0
python-docx>=1.1.0
markdown2>=2.5.0
markdownify>=0.13.0
pydub>=0.25.1
ffmpeg-python>=0.2.0
kokoro>=0.9.0
TTS>=0.22.0
google-api-python-client>=2.130.0
google-auth-httplib2>=0.2.0
google-auth-oauthlib>=1.2.0
platformdirs>=4.2.0
```

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve

# Launch
python main.py
```

---

## Environment Variables

| Variable        | Default                    | Description                        |
|-----------------|----------------------------|------------------------------------|
| `OLLAMA_HOST`   | `http://localhost:11434`   | Ollama API base URL                |
| `TTS_ENGINE`    | `kokoro`                   | Primary TTS engine (`kokoro`/`coqui`) |
| `VERITAS_DEBUG` | `0`                        | Set to `1` for verbose logging     |

---

## Known Constraints & Decisions

- **Kokoro-82M** is preferred for TTS because of audio quality; it requires a local model download. Coqui is the fallback for voice variety.
- PyQt6 was chosen over CustomTkinter for: native OS look, `QMediaPlayer` for audio playback, `QThreadPool` for async workers, and better long-term maintenance.
- Silence removal uses `pydub` (pure Python, no extra deps beyond ffmpeg binary) rather than a neural VAD for simplicity and reliability.
- Google Docs sync is optional — the app functions fully offline without it.
