"""Event collector that merges enterprise events from multiple sources."""

from __future__ import annotations

import json
from pathlib import Path

from src.proactive.event_models import EnterpriseEvent, EventType, Severity

EVENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "events"

EVENT_FILES: dict[str, EventType] = {
    "training_events.json": EventType.TRAINING_EXPIRY,
    "passport_events.json": EventType.PASSPORT_EXPIRY,
    "expense_events.json": EventType.EXPENSE_BLOCKED,
    "leave_events.json": EventType.LEAVE_CONFLICT,
    "travel_events.json": EventType.PENDING_APPROVAL,
    "deployment_events.json": EventType.DEPLOYMENT_CONFLICT,
    "approval_events.json": EventType.PENDING_APPROVAL,
    "warranty_events.json": EventType.WARRANTY_EXPIRY,
}


class EventCollector:
    """Collects and merges events from JSON data files."""

    def __init__(self, events_dir: Path | None = None):
        self.events_dir = events_dir or EVENTS_DIR

    def _load_file(self, filename: str) -> list[dict]:
        filepath = self.events_dir / filename
        if not filepath.exists():
            return []
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []

    def collect_all(self) -> list[EnterpriseEvent]:
        """Load all events from every JSON file and return merged list."""
        all_events: list[EnterpriseEvent] = []
        for filename, default_type in EVENT_FILES.items():
            raw_events = self._load_file(filename)
            for raw in raw_events:
                try:
                    event = EnterpriseEvent.from_dict(raw)
                except (ValueError, KeyError):
                    event = EnterpriseEvent(
                        event_id=raw.get("event_id", ""),
                        employee_id=raw.get("employee_id", ""),
                        employee_name=raw.get("employee_name", ""),
                        department=raw.get("department", ""),
                        event_type=default_type,
                        severity=Severity(raw.get("severity", "Medium")),
                        source_system=raw.get("source_system", ""),
                        title=raw.get("title", ""),
                        description=raw.get("description", ""),
                        metadata=raw.get("metadata", {}),
                    )
                all_events.append(event)
        return all_events

    def collect_for_employee(self, employee_id: str) -> list[EnterpriseEvent]:
        """Return events relevant to a specific employee."""
        return [e for e in self.collect_all() if e.employee_id == employee_id]

    def collect_by_type(self, event_type: EventType) -> list[EnterpriseEvent]:
        """Return events of a specific type."""
        return [e for e in self.collect_all() if e.event_type == event_type]

    def collect_by_severity(self, severity: Severity) -> list[EnterpriseEvent]:
        """Return events of a specific severity."""
        return [e for e in self.collect_all() if e.severity == severity]
