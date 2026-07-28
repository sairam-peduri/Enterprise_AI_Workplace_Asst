"""Notification manager that prevents recommendation overload."""

from __future__ import annotations

from datetime import datetime

from src.proactive.event_models import Recommendation, Severity

MAX_RECOMMENDATIONS_PER_SESSION = 8
DEDUP_WINDOW_HOURS = 24


class NotificationManager:
    """Manages notification delivery, deduplication, and throttling."""

    def __init__(self, max_per_session: int = MAX_RECOMMENDATIONS_PER_SESSION):
        self.max_per_session = max_per_session
        self._history: list[dict] = []
        self._delivered_ids: set[str] = set()
        self._suppressed_count: int = 0

    def filter_recommendations(self, recommendations: list[Recommendation], employee_id: str | None = None) -> list[Recommendation]:
        """Filter and limit recommendations for delivery."""
        candidates = recommendations
        if employee_id:
            candidates = [r for r in candidates if r.employee_id == employee_id]
        candidates = self._deduplicate(candidates)
        candidates = self._suppress_recent(candidates)
        candidates = self._prioritize(candidates)
        delivered = candidates[: self.max_per_session]
        self._suppressed_count += len(candidates) - len(delivered)
        for rec in delivered:
            self._delivered_ids.add(rec.recommendation_id)
            self._history.append({
                "recommendation_id": rec.recommendation_id,
                "title": rec.title,
                "priority": rec.priority.value,
                "employee_id": rec.employee_id,
                "timestamp": datetime.now().isoformat(),
                "action": "delivered",
            })
        return delivered

    def _deduplicate(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        seen_titles: set[str] = set()
        unique: list[Recommendation] = []
        for rec in recommendations:
            key = f"{rec.employee_id}:{rec.title.lower().strip()}"
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(rec)
        return unique

    def _suppress_recent(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        return [r for r in recommendations if r.recommendation_id not in self._delivered_ids]

    def _prioritize(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        priority_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        return sorted(recommendations, key=lambda r: (priority_order.get(r.priority, 4), -r.confidence))

    def dismiss_recommendation(self, recommendation_id: str) -> None:
        self._history.append({
            "recommendation_id": recommendation_id,
            "timestamp": datetime.now().isoformat(),
            "action": "dismissed",
        })

    def accept_recommendation(self, recommendation_id: str) -> None:
        self._history.append({
            "recommendation_id": recommendation_id,
            "timestamp": datetime.now().isoformat(),
            "action": "accepted",
        })

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    @property
    def history(self) -> list[dict]:
        return list(self._history)
