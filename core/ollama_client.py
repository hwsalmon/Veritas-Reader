"""Ollama HTTP API client.

Communicates with a locally running Ollama server.
Base URL is read from the OLLAMA_HOST environment variable,
defaulting to http://localhost:11434.
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or is unreachable."""


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_HOST) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return a sorted list of installed model names (e.g. 'llama3.2:latest')."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return sorted(m["name"] for m in data.get("models", []))
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama API error: {exc}") from exc

    def is_running(self) -> bool:
        """Quick health-check — returns True if Ollama is reachable."""
        try:
            self._session.get(f"{self.base_url}/", timeout=2).raise_for_status()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Text generation (streaming)
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[str]:
        """Yield response tokens from /api/generate.

        Args:
            model: Model name as returned by list_models().
            prompt: User prompt text.
            system: Optional system message.
            temperature: Sampling temperature (0.0–2.0).
            stream: If True, yields tokens as they arrive; if False, yields a
                    single complete response string.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=(5, 120),
            ) as resp:
                resp.raise_for_status()
                if stream:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                else:
                    yield resp.json().get("response", "")
        except requests.ConnectionError as exc:
            raise OllamaError(f"Connection lost during generation: {exc}") from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama generation error: {exc}") from exc

    # ------------------------------------------------------------------
    # Chat completion (streaming)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, model: str, text: str) -> list[float]:
        """Return a dense embedding vector for the given text.

        Tries the current /api/embed endpoint first, falls back to the
        legacy /api/embeddings endpoint for older Ollama versions.

        Args:
            model: Embedding model name (e.g. 'nomic-embed-text').
            text: Text to embed.
        """
        try:
            # Current Ollama API (≥0.1.26)
            resp = self._session.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=30,
            )
            if resp.status_code == 404:
                # Fall back to legacy endpoint
                resp = self._session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            resp.raise_for_status()
            # New API returns {"embeddings": [[...]]}
            return resp.json()["embeddings"][0]
        except requests.ConnectionError as exc:
            raise OllamaError(f"Connection lost during embedding: {exc}") from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama embedding error: {exc}") from exc

    # ------------------------------------------------------------------
    # Chat completion (streaming)
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[str]:
        """Yield response tokens from /api/chat.

        Args:
            model: Model name.
            messages: List of {"role": "user"|"assistant"|"system", "content": str}.
            temperature: Sampling temperature.
            stream: Stream tokens if True.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        try:
            with self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=(5, 120),
            ) as resp:
                resp.raise_for_status()
                if stream:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                else:
                    yield resp.json().get("message", {}).get("content", "")
        except requests.ConnectionError as exc:
            raise OllamaError(f"Connection lost during chat: {exc}") from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama chat error: {exc}") from exc
