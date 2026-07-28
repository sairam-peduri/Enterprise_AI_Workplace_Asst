"""Approval manager that handles user confirmation before executing actions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.proactive.event_models import Recommendation


class ApprovalManager:
    """Manages the approval workflow for proactive recommendations."""

    def __init__(self):
        self._pending: dict[str, dict[str, Any]] = {}
        self._approval_history: list[dict[str, Any]] = []

    def request_approval(self, recommendation: Recommendation) -> dict[str, Any]:
        """Create an approval request from a recommendation."""
        request = {
            "recommendation_id": recommendation.recommendation_id,
            "employee_id": recommendation.employee_id,
            "title": recommendation.title,
            "suggested_action": recommendation.suggested_action,
            "approval_type": recommendation.approval_type,
            "correlated_events": [e.to_dict() for e in recommendation.correlated_events],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        self._pending[recommendation.recommendation_id] = request
        return request

    def get_approval_message(self, recommendation: Recommendation) -> str:
        """Generate the human-readable approval prompt."""
        lines = [f"**{recommendation.title}**\n"]
        lines.append(f"Reason: {recommendation.reason}\n")
        lines.append(f"Business Impact: {recommendation.business_impact}\n")
        lines.append(f"Suggested Action: {recommendation.suggested_action}\n")
        if recommendation.approval_required:
            lines.append("\nWould you like me to:")
            lines.append(f"- {recommendation.suggested_action}")
            lines.append("\nReply **yes** to approve, **no** to dismiss.")
        else:
            lines.append("\nThis is an informational recommendation. No action required.")
        return "\n".join(lines)

    def approve(self, recommendation_id: str) -> bool:
        """Record user approval."""
        if recommendation_id in self._pending:
            self._pending[recommendation_id]["status"] = "approved"
            self._pending[recommendation_id]["approved_at"] = datetime.now().isoformat()
            self._approval_history.append(self._pending.pop(recommendation_id))
            return True
        return False

    def dismiss(self, recommendation_id: str) -> bool:
        """Record user dismissal."""
        if recommendation_id in self._pending:
            self._pending[recommendation_id]["status"] = "dismissed"
            self._pending[recommendation_id]["dismissed_at"] = datetime.now().isoformat()
            self._approval_history.append(self._pending.pop(recommendation_id))
            return True
        return False

    def is_pending(self, recommendation_id: str) -> bool:
        return recommendation_id in self._pending

    def get_pending(self) -> list[dict[str, Any]]:
        return list(self._pending.values())

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._approval_history)
