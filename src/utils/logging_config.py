"""Structured logging and per-request activity tracing for Enterprise AI."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator


LOG_DIRECTORY = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE = LOG_DIRECTORY / "enterprise_ai.log"
LOGGER_NAME = "enterprise_ai"
_activity_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "enterprise_ai_activity_events", default=None
)


def get_logger() -> logging.Logger:
    """Return the configured application logger without duplicating handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


@contextmanager
def capture_activity() -> Iterator[list[dict[str, Any]]]:
    """Capture structured events emitted during one chat request."""
    events: list[dict[str, Any]] = []
    token = _activity_events.set(events)
    try:
        yield events
    finally:
        _activity_events.reset(token)


def record_event(event: str, **details: Any) -> None:
    """Write an event to the log file and the active request trace, if any."""
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **details,
    }
    get_logger().info(json.dumps(payload, default=str, sort_keys=True))
    events = _activity_events.get()
    if events is not None:
        events.append(payload)
