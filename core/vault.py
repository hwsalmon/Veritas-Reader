"""Per-document vault — organises all files associated with one document.

Layout::

    vault_root/
      <stem>/
        versions/       ← autosaves and committed versions
        ai_history/     ← AI chat sessions (JSON)
        audio/          ← TTS raw + processed WAV files
        kb/             ← knowledge base (.vkb) files
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Vault:
    """Directory hierarchy for one document's associated files."""

    def __init__(self, vault_root: Path, stem: str) -> None:
        self.stem = stem
        self.root = vault_root / stem
        self.versions_dir = self.root / "versions"
        self.ai_history_dir = self.root / "ai_history"
        self.audio_dir = self.root / "audio"
        self.kb_dir = self.root / "kb"

    def init(self) -> None:
        """Create all subdirectories (idempotent)."""
        for d in [self.versions_dir, self.ai_history_dir, self.audio_dir, self.kb_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Vault ready: %s", self.root)

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def next_version_path(self, suffix: str = ".md") -> Path:
        """Return a timestamped v_YYYYMMDD_HHMMSS path inside versions/.

        Falls back to appending _2, _3 … if the same second fires twice.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.versions_dir / f"v_{ts}{suffix}"
        if not candidate.exists():
            return candidate
        n = 2
        while True:
            candidate = self.versions_dir / f"v_{ts}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def list_versions(self) -> list[Path]:
        """Return version files sorted oldest-first (both legacy stem-N and new v_ naming)."""
        paths: list[Path] = []
        for ext in (".md", ".txt", ".docx"):
            paths.extend(self.versions_dir.glob(f"{self.stem}-*{ext}"))  # legacy
            paths.extend(self.versions_dir.glob(f"v_*{ext}"))            # timestamped
        return sorted(set(paths))

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def audio_path(self, name: str) -> Path:
        return self.audio_dir / name

    # ------------------------------------------------------------------
    # AI chat history
    # ------------------------------------------------------------------

    def save_ai_session(
        self,
        session_type: str,
        model: str,
        messages: list[dict],
        document_name: str = "",
    ) -> Path:
        """Persist a conversation as JSON and return the saved path."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}_{session_type}.json"
        path = self.ai_history_dir / filename
        data = {
            "type": session_type,
            "model": model,
            "document": document_name,
            "timestamp": ts,
            "messages": messages,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("AI session saved: %s", path.name)
        return path

    def list_ai_sessions(self) -> list[Path]:
        """Return session JSON files sorted newest-first."""
        return sorted(self.ai_history_dir.glob("*.json"), reverse=True)

    def load_ai_session(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------

    def default_kb_path(self) -> Path:
        return self.kb_dir / f"{self.stem}.vkb"

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    @property
    def session_path(self) -> Path:
        """Path to the workspace session file for this project."""
        return self.root / "session.json"

    def save_session(self, data: dict) -> None:
        """Persist workspace state to session.json using atomic write."""
        tmp = self.session_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.session_path)
            logger.debug("Session saved: %s", self.session_path)
        except Exception as exc:
            logger.warning("Failed to save session: %s", exc)
            tmp.unlink(missing_ok=True)

    def load_session(self) -> dict | None:
        """Load session.json, returning None if absent or corrupt."""
        if not self.session_path.exists():
            return None
        try:
            return json.loads(self.session_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load session.json: %s", exc)
            return None
