"""Session persistence utilities for saving/loading chat sessions to disk."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

SESSIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "sessions.json"


def _ensure_data_dir():
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _serialize_message(msg) -> dict:
    """Convert a LangChain message to a serializable dict."""
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": msg.content}
    if isinstance(msg, AIMessage):
        return {
            "type": "ai",
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs,
        }
    return {"type": "unknown", "content": str(msg)}


def _deserialize_message(data: dict):
    """Convert a dict back to a LangChain message."""
    msg_type = data.get("type")
    if msg_type == "human":
        return HumanMessage(content=data["content"])
    if msg_type == "ai":
        return AIMessage(
            content=data["content"],
            additional_kwargs=data.get("additional_kwargs", {}),
        )
    return HumanMessage(content=data.get("content", ""))


def load_sessions() -> dict:
    """Load all saved sessions from disk."""
    _ensure_data_dir()
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for session in raw.values():
            session["messages"] = [
                _deserialize_message(m) for m in session.get("messages", [])
            ]
        return raw
    except (json.JSONDecodeError, IOError):
        return {}


def save_sessions(sessions: dict) -> None:
    """Save all sessions to disk, converting LangChain messages to dicts."""
    _ensure_data_dir()
    serializable = {}
    for sid, session in sessions.items():
        s = dict(session)
        s["messages"] = [_serialize_message(m) for m in s.get("messages", [])]
        serializable[sid] = s
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
