# Scriptum Veritas

> A local-first desktop writing environment — import documents, refine with AI, synthesise to speech, export everything.

Scriptum Veritas combines a rich markdown editor, local LLM text generation via [Ollama](https://ollama.com), high-fidelity text-to-speech via Kokoro-82M, and an embedded browser panel — all in one offline-capable desktop application built with Python and PyQt6.

---

## Features

### File Management
- **New File** (`Ctrl+N`) — clear the editor with a Save / Discard / Cancel prompt when there are unsaved changes
- **Import Documents** — multi-select `.md`, `.txt`, or `.docx` files; copies them into `VeritasVault/imports/` with automatic collision avoidance
- **Open from Imports** — browse the imports folder in a dialog, double-click to load; includes a "Reveal Folder" button
- **Close All Tabs** — closes every scratch tab (prompts on unsaved content), leaving the primary document tab intact
- **Exit** (`Ctrl+Q`) — autosaves then closes cleanly

### Editor
- Multi-tab editor with clone, right-click close, and unsaved-content warnings
- Markup ↔ Formatted toggle — switch between raw markdown and rendered rich text
- Style toolbar: headings (H1–H3), bold, italic, strikethrough, blockquote, bullet list, indent/outdent
- Find & Replace with match-case, whole-word, and replace-all
- US-English spell checking with custom user dictionary and inline suggestions
- AI-powered grammar check (feeds editor text directly into the Generate panel)
- Versioned autosave every 60 seconds into the active Vault

### AI Generation
- Chat with any locally-running Ollama model — streamed token-by-token
- Follow-up conversation mode — continue a generation without re-entering context
- Knowledge Base (RAG) — embed your document with `nomic-embed-text`, then ask questions over it using cosine-similarity retrieval
- AI session history — every generate/KB-chat session saved to the Vault and reloadable
- One-click "Open in New Tab" to move AI output into a fresh editor tab

### Text-to-Speech
- Primary engine: **Kokoro-82M ONNX** — high quality, low latency
- Per-segment pacing: speed, sentence pause, m-dash pause, paragraph pause
- 5-stage audio pipeline: silence removal → EQ → soft-knee compression → loudness normalisation → 44.1 kHz upsample
- Export as `.wav` (lossless) or `.mp3` (192k)

### Embedded Browsers
- Persistent browser tabs backed by a single `QWebEngineProfile` (cookies survive restarts)
- Quick-launch buttons for **Substack**, **NotebookLM**, **ChatGPT**, **Gemini**, and **Claude AI** — log in once, stay logged in

### Vault System
- Every document lives in `~/Documents/VeritasVault/<stem>/` with subfolders for `versions/`, `ai_history/`, `audio/`, and `kb/`
- "Commit Version" and Clone Tab both write a new numbered snapshot
- Choose Vault root from the File menu

---

## Screenshots

> *Coming soon*

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.14 (Fedora) |
| [Ollama](https://ollama.com) | Running locally on `http://localhost:11434` |
| FFmpeg | For audio export (`ffmpeg` binary in PATH) |
| Wayland or X11 | Wayland recommended on Linux |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/hwsalmon/scriptum-veritas.git
cd scriptum-veritas

# 2. Create virtual environment (uv recommended)
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull llama3.2          # or any model you prefer
ollama pull nomic-embed-text  # required for Knowledge Base / RAG

# 5. Download Kokoro ONNX model files
#    Place in ~/.local/share/veritas_reader/models/
#    Required files:
#      kokoro-v1.0.fp16.onnx  (170 MB)
#      voices-v1.0.bin        (27 MB)
#    Available at: https://huggingface.co/hexgrad/Kokoro-82M
```

---

## Running

```bash
# Wayland
QT_QPA_PLATFORM=wayland .venv/bin/python main.py

# X11
.venv/bin/python main.py
```

Or, if you have installed the `.desktop` entry, launch **Scriptum Veritas** from your application menu.

---

## Project Structure

```
scriptum-veritas/
├── main.py                    # Entry point
├── app/
│   ├── window.py              # MainWindow
│   ├── editor.py              # Markup/formatted editor widget
│   ├── editor_tabs.py         # Multi-tab container with overlay buttons
│   ├── web_tab.py             # Persistent embedded browser (QWebEngineView)
│   ├── player.py              # Audio player (QMediaPlayer)
│   ├── toolbar.py             # ModelSelector, VoiceSelector, PacingControls
│   ├── theme.py               # Dark / light palette
│   └── dialogs/
│       ├── paste_dialog.py
│       ├── gdocs_dialog.py
│       └── imports_dialog.py  # Imports folder browser
├── core/
│   ├── file_handler.py        # .md / .txt / .docx read & write
│   ├── ollama_client.py       # Ollama HTTP client (generate, chat, embed)
│   ├── tts_engine.py          # Kokoro ONNX TTS engine
│   ├── audio_processor.py     # 5-stage audio pipeline
│   ├── knowledge_base.py      # Chunking, embedding, cosine retrieval
│   ├── text_preprocessor.py   # Markdown stripping + prosodic expansion
│   ├── vault.py               # Versioned document vault
│   └── gdocs.py               # Google Docs OAuth integration
├── config/
│   └── settings.py            # QSettings-backed app preferences
└── assets/
    └── icons/                 # App icon (SVG + PNG)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `TTS_ENGINE` | `kokoro` | TTS engine (`kokoro` / `coqui`) |
| `ONNX_PROVIDER` | *(auto)* | Override ONNX execution provider |
| `VERITAS_DEBUG` | `0` | Set to `1` for verbose logging |

---

## Google Docs Integration (optional)

1. Create a Google Cloud project and enable the Docs + Drive APIs
2. Download OAuth credentials as `config/google_oauth.json`
3. On first use, a browser window will open for authentication
4. Token stored at `~/.config/veritas_reader/token.json`

---

## Tech Stack

| Concern | Library |
|---|---|
| GUI | PyQt6 |
| Embedded browser | PyQt6-WebEngine |
| Local LLM | Ollama (HTTP API) |
| TTS | kokoro-onnx (Kokoro-82M) |
| Audio post-processing | pydub, scipy, numpy |
| Document parsing | python-docx, markdown2, markdownify |
| Spell check | pyspellchecker |
| Embeddings / RAG | Ollama + nomic-embed-text |
| Google Docs | google-api-python-client, google-auth-oauthlib |

---

## License

MIT
