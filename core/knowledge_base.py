"""Knowledge base: chunk, embed, persist, and query documents via RAG."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_CHUNK_CHARS = 700
MIN_CHUNK_CHARS = 50
DEFAULT_TOP_K = 6


@dataclass
class KnowledgeBase:
    """A named collection of text chunks with their embeddings."""

    name: str
    embedding_model: str
    chunks: list[str]
    embeddings: np.ndarray          # shape (N, D), float32
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def query(self, question_embedding: list[float], top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Return the top_k chunks most semantically similar to the question."""
        if not self.chunks:
            return []
        q = np.array(question_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = self.embeddings / norms
        scores = normed @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "created": self.created,
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.debug("Saved KB '%s' → %s (%d chunks)", self.name, path, len(self.chunks))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            embedding_model=data["embedding_model"],
            created=data.get("created", ""),
            chunks=data["chunks"],
            embeddings=np.array(data["embeddings"], dtype=np.float32),
        )


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into semantic chunks by paragraph, sub-splitting long ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            if len(para) >= MIN_CHUNK_CHARS:
                chunks.append(para)
        else:
            # Sub-split on sentence boundaries
            for sep in (". ", "? ", "! "):
                para = para.replace(sep, sep[:-1] + "|")
            sentences = para.split("|")
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if len(current) >= MIN_CHUNK_CHARS:
                        chunks.append(current)
                    current = sent
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)

    return chunks


def build_rag_system_prompt(chunks: list[str]) -> str:
    """Build a system prompt that positions the model as a writing coach/critic."""
    excerpts = "\n---\n".join(chunks)
    return (
        "You are an insightful writing coach and literary critic. The user is asking "
        "about their own writing. The excerpts below are drawn from their work — use "
        "them as your primary evidence. You may analyse, critique, and form opinions "
        "freely: discuss structure, argumentation, style, strengths, weaknesses, and "
        "suggest improvements. Quote specific passages to support your points. If a "
        "question requires knowledge beyond the excerpts (e.g. broad structural "
        "observations), draw reasonable inferences from what you can see.\n\n"
        f"Excerpts from the work:\n---\n{excerpts}\n---"
    )
