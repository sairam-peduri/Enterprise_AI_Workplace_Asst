"""Memory manager that tracks recommendation history to prevent repetition."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_FILE = Path(__file__).resolve().parents[2] / "data" / "proactive_memory.json"


class MemoryManager:
    """Stores past recommendations and prevents repeated notifications."""

    def __init__(self, memory_file: Path | None = None):
        self.memory_file = memory_file or MEMORY_FILE
        self._memory = self._load()

    def _load(self) -> dict[str, Any]:
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {"dismissed": [], "accepted": [], "executed": [], "timestamps": {}}

    def _save(self) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps(self._memory, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_dismissed(self, recommendation_id: str, employee_id: str, title: str) -> None:
        entry = {"id": recommendation_id, "employee_id": employee_id, "title": title, "timestamp": datetime.now().isoformat()}
        if entry not in self._memory["dismissed"]:
            self._memory["dismissed"].append(entry)
            self._save()

    def record_accepted(self, recommendation_id: str, employee_id: str, title: str) -> None:
        entry = {"id": recommendation_id, "employee_id": employee_id, "title": title, "timestamp": datetime.now().isoformat()}
        if entry not in self._memory["accepted"]:
            self._memory["accepted"].append(entry)
            self._save()

    def record_executed(self, recommendation_id: str, employee_id: str, title: str, action: str) -> None:
        entry = {"id": recommendation_id, "employee_id": employee_id, "title": title, "action": action, "timestamp": datetime.now().isoformat()}
        self._memory["executed"].append(entry)
        self._save()

    def is_dismissed(self, recommendation_id: str) -> bool:
        return any(d["id"] == recommendation_id for d in self._memory["dismissed"])

    def was_similar_title_dismissed(self, employee_id: str, title: str) -> bool:
        title_lower = title.lower().strip()
        return any(
            d["employee_id"] == employee_id and d["title"].lower().strip() == title_lower
            for d in self._memory["dismissed"]
        )

    def get_employee_history(self, employee_id: str) -> dict[str, list]:
        return {
            "dismissed": [d for d in self._memory["dismissed"] if d["employee_id"] == employee_id],
            "accepted": [a for a in self._memory["accepted"] if a["employee_id"] == employee_id],
            "executed": [e for e in self._memory["executed"] if e["employee_id"] == employee_id],
        }

    def update_timestamp(self, employee_id: str) -> None:
        self._memory["timestamps"][employee_id] = datetime.now().isoformat()
        self._save()

    def get_last_check(self, employee_id: str) -> str | None:
        return self._memory["timestamps"].get(employee_id)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "dismissed": len(self._memory["dismissed"]),
            "accepted": len(self._memory["accepted"]),
            "executed": len(self._memory["executed"]),
        }
