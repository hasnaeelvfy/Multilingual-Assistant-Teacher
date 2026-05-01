"""
Local conversation memory for the assistant.

- Embeddings: sentence-transformers (runs on your machine, no API key).
- Similarity search: FAISS (CPU, open source). Inner product on L2-normalized
  vectors equals cosine similarity.
- Persistence: FAISS index + JSON metadata under conversation_memory_store/.
- Cross-language retrieval: user/assistant text and search queries are translated
  to English (deep-translator, free) before embedding so FAISS stays in one space.

Disable with env: CONVERSATION_MEMORY=0
Override embedding model with: SENTENCE_TRANSFORMER_MODEL=...
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_DIR = ROOT / "conversation_memory_store"

ENABLED = os.getenv("CONVERSATION_MEMORY", "1").strip().lower() not in {"0", "false", "no", "off"}

_MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _norm_lang(code: str) -> str:
    return (code or "en").strip().lower().split("-")[0]


def _to_english(text: str) -> str:
    """Translate to English for embedding; on failure returns stripped original."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source="auto", target="en").translate(text)
        return (out or text).strip()
    except Exception:
        return text


class ConversationMemory:
    """
    Stores each turn as one embedding of "User: ...\\nAssistant: ..." (both in
    English). The user side uses the LLM ``corrected`` text when provided so noisy
    STT does not poison translation/embeddings. ``selected_language`` on each record
    is metadata only (e.g. for debugging).
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._store_dir = Path(store_dir or DEFAULT_STORE_DIR)
        self._index_path = self._store_dir / "vectors.faiss"
        self._meta_path = self._store_dir / "metadata.json"
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._index = None
        self._records: list[dict[str, Any]] = []
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_MODEL_NAME)

    def _embedding_dim(self) -> int:
        self._ensure_model()
        if self._dim is not None:
            return self._dim
        probe = self._model.encode(["dimension"], convert_to_numpy=True, show_progress_bar=False)
        self._dim = int(np.asarray(probe).shape[-1])
        return self._dim

    def _new_index(self):
        import faiss

        return faiss.IndexFlatIP(self._embedding_dim())

    def _encode(self, texts: list[str]) -> np.ndarray:
        import faiss

        self._ensure_model()
        emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        x = np.asarray(emb, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        faiss.normalize_L2(x)
        return x

    def load(self) -> None:
        """Load index + metadata from disk, or start empty."""
        import faiss

        self._index = self._new_index()
        self._records = []
        if not self._meta_path.exists() or not self._index_path.exists():
            return
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            records = meta.get("records") or []
            stored_model = meta.get("model", _MODEL_NAME)
            if stored_model != _MODEL_NAME:
                print(f"[memory] embedding model changed ({stored_model!r} -> {_MODEL_NAME!r}); ignoring old index")
                return
            idx = faiss.read_index(str(self._index_path))
            if idx.ntotal != len(records):
                print(f"[memory] index size {idx.ntotal} != records {len(records)}; starting fresh")
                return
            if hasattr(idx, "d") and idx.d != self._embedding_dim():
                print("[memory] vector dimension mismatch; starting fresh")
                return
            self._index = idx
            self._records = list(records)
        except Exception as e:
            print(f"[memory] could not load store: {e}; starting fresh")
            self._index = self._new_index()
            self._records = []

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def _save(self) -> None:
        import faiss

        self._store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        payload = {
            "records": self._records,
            "dim": self._embedding_dim(),
            "model": _MODEL_NAME,
        }
        self._meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_turn(
        self,
        *,
        user_text: str,
        answer: str,
        selected_language: str,
        corrected: str = "",
        explanation: str = "",
    ) -> None:
        """Append one exchange and persist. Skips empty user or empty answer."""
        if not ENABLED:
            return
        user_text = (user_text or "").strip()
        answer = (answer or "").strip()
        if not user_text or not answer:
            return
        # Prefer LLM-corrected wording for translation + embedding (raw STT often mistranslates).
        user_for_embed = (corrected or "").strip() or user_text
        user_en = _to_english(user_for_embed) or user_for_embed
        answer_en = _to_english(answer) or answer
        combined = f"User: {user_en}\nAssistant: {answer_en}"
        with self._lock:
            self._ensure_loaded()
            vec = self._encode([combined])
            self._index.add(vec)
            self._records.append(
                {
                    "user_text": user_en,
                    "answer": answer_en,
                    "corrected": (corrected or "").strip(),
                    "explanation": (explanation or "").strip(),
                    "selected_language": _norm_lang(selected_language),
                }
            )
            self._save()

    def search(self, query: str, *, top_k: int = 3, selected_language: str | None = None) -> list[dict[str, Any]]:
        """
        Return up to top_k past turns most similar to the current user message
        (cosine similarity on embeddings). The query is translated to English
        before search so it aligns with English-normalized index vectors.
        ``selected_language`` is accepted for backward compatibility and ignored.
        """
        _ = selected_language  # unused; kept so older call sites keep working
        if not ENABLED:
            return []
        query = (query or "").strip()
        if not query:
            return []
        query_en = _to_english(query) or query
        with self._lock:
            self._ensure_loaded()
            if self._index is None or self._index.ntotal == 0:
                return []
            q = self._encode([query_en])
            n_fetch = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(q, n_fetch)

        out: list[dict[str, Any]] = []
        for idx in ids[0].tolist():
            if idx < 0 or idx >= len(self._records):
                continue
            out.append(dict(self._records[idx]))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def format_for_prompt(records: list[dict[str, Any]]) -> str:
        """Turn retrieved turns into a short plain-text block for the system prompt."""
        if not records:
            return ""
        lines: list[str] = []
        for i, r in enumerate(records, 1):
            u = (r.get("user_text") or "").strip()
            a = (r.get("answer") or "").strip()
            lines.append(f"{i}) User: {u}\n   Assistant: {a}")
        return "\n".join(lines)


_singleton: ConversationMemory | None = None
_singleton_lock = threading.Lock()


def get_conversation_memory() -> ConversationMemory:
    """Process-wide singleton so all requests share the same index and files."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = ConversationMemory()
        return _singleton
