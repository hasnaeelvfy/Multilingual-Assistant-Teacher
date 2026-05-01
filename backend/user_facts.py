"""
Persistent user facts (name, simple preferences) extracted from user messages.

Embeddings alone often miss cross-language recall (e.g. English introduction +
French \"what is my name?\"). Facts are regex-extracted, saved locally, and
always prepended to the tutor memory block so they appear in every system prompt.

Stored next to vector memory: conversation_memory_store/user_facts.json
Disable with: USER_FACTS_MEMORY=0
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = ROOT / "conversation_memory_store" / "user_facts.json"

ENABLED = os.getenv("USER_FACTS_MEMORY", "1").strip().lower() not in {"0", "false", "no", "off"}

_LANG_LABEL = {"en": "English", "fr": "French", "ar": "Arabic", "es": "Spanish"}

_lock = threading.Lock()

_MAX_LIKES = 8
_MAX_NAME_LEN = 60


@dataclass
class UserFacts:
    name: str | None = None
    likes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "likes": list(self.likes)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> UserFacts:
        name = data.get("name")
        likes = data.get("likes") or []
        if not isinstance(likes, list):
            likes = []
        likes = [str(x).strip() for x in likes if str(x).strip()]
        return cls(name=str(name).strip() if name else None, likes=likes[:_MAX_LIKES])


def _norm_lang(code: str) -> str:
    return (code or "en").strip().lower().split("-")[0]


def _lang_label(code: str) -> str:
    return _LANG_LABEL.get(_norm_lang(code), "the selected teaching language")


def load_facts() -> UserFacts:
    if not FACTS_PATH.exists():
        return UserFacts()
    try:
        data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return UserFacts()
        return UserFacts.from_json(data)
    except Exception:
        return UserFacts()


def save_facts(facts: UserFacts) -> None:
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.write_text(json.dumps(facts.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_name_candidate(raw: str) -> str:
    s = (raw or "").strip().strip('\'"”“«»')
    s = re.split(r"[,;]|\.(?=\s|$)|!|\?", s)[0].strip()
    s = re.sub(r"\s+", " ", s)
    if not s or len(s) > _MAX_NAME_LEN:
        return ""
    return s


def extract_name_from_text(text: str) -> str | None:
    """Conservative name extraction (multilingual)."""
    t = (text or "").strip()
    if not t:
        return None

    patterns = [
        r"(?i)my\s+name\s+is\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"(?i)my\s+name's\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"(?i)call\s+me\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"(?i)i(?:'m|\s+am)\s+called\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"(?i)je\s+m['’]appelle\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"(?i)mon\s+nom\s+est\s+([^\n.!?]{1,%d})" % _MAX_NAME_LEN,
        r"اسمي\s+([^\n.!?،]{1,%d})" % _MAX_NAME_LEN,
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            name = _strip_name_candidate(m.group(1))
            if name and len(name) >= 2:
                return name
    return None


def extract_like_from_text(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    patterns = [
        r"(?i)\b(?:i\s+like|i\s+love)\s+([^.!?\n]{1,50})",
        r"(?i)\bj'aime\s+([^.!?\n]{1,50})",
        r"(?i)\bj'adore\s+([^.!?\n]{1,50})",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            like = _strip_name_candidate(m.group(1))
            if like and len(like) >= 2:
                return like[:80]
    return None


def ingest_from_user_text(text: str) -> None:
    """Update stored facts from a single user utterance (after normalization)."""
    if not ENABLED:
        return
    text = (text or "").strip()
    if not text:
        return

    with _lock:
        facts = load_facts()
        changed = False

        name = extract_name_from_text(text)
        if name:
            facts.name = name
            changed = True

        like = extract_like_from_text(text)
        if like:
            key = like.lower()
            existing = {x.lower() for x in facts.likes}
            if key not in existing and len(facts.likes) < _MAX_LIKES:
                facts.likes.append(like)
                changed = True

        if changed:
            save_facts(facts)


def facts_block_for_prompt(*, selected_language: str) -> str:
    """
    Text block always injected before semantic memory turns.
    Instructs model to use facts naturally while keeping JSON in the teaching language.
    """
    if not ENABLED:
        return ""
    facts = load_facts()
    label = _lang_label(selected_language)
    lines: list[str] = []

    if facts.name:
        lines.append(f"- The user's preferred name is: {facts.name}.")
    if facts.likes:
        likes_txt = "; ".join(facts.likes[:_MAX_LIKES])
        lines.append(f"- They said they like: {likes_txt}.")

    if not lines:
        return ""

    return (
        "Persistent facts about this user (from earlier messages, possibly in another language). "
        "Use them when relevant (e.g. recalling their name). "
        f"All JSON fields must still be written entirely in {label} only:\n"
        + "\n".join(lines)
    )


def merge_memory_sections(*, facts_block: str, semantic_block: str) -> str:
    parts = [p.strip() for p in (facts_block, semantic_block) if p and p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return (
        parts[0]
        + "\n\nEarlier relevant conversation turns (semantic recall; any language):\n"
        + parts[1]
    )
