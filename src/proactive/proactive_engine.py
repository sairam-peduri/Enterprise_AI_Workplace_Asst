"""Main proactive engine that orchestrates the full proactive intelligence pipeline."""

from __future__ import annotations

from typing import Any

from src.proactive.event_collector import EventCollector
from src.proactive.event_correlator import EventCorrelator
from src.proactive.priority_engine import PriorityEngine
from src.proactive.recommendation_engine import RecommendationEngine
from src.proactive.notification_manager import NotificationManager
from src.proactive.context_manager import ContextManager
from src.proactive.approval_manager import ApprovalManager
from src.proactive.memory_manager import MemoryManager
from src.proactive.event_models import Recommendation, Severity


class ProactiveEngine:
    """Orchestrates the full proactive pipeline: collect → correlate → prioritize → recommend → notify."""

    def __init__(self):
        self.collector = EventCollector()
        self.correlator = EventCorrelator()
        self.priority_engine = PriorityEngine()
        self.recommendation_engine = RecommendationEngine()
        self.notification_manager = NotificationManager()
        self.context_manager = ContextManager()
        self.approval_manager = ApprovalManager()
        self.memory_manager = MemoryManager()

    def run_pipeline(self, employee_id: str | None = None) -> list[Recommendation]:
        """Execute the full proactive pipeline and return actionable recommendations."""
        all_events = self.collector.collect_all()
        correlated = self.correlator.correlate(all_events)
        recommendations = self.recommendation_engine.generate_batch(correlated)
        if employee_id:
            recommendations = [r for r in recommendations if r.employee_id == employee_id]
        filtered = self.notification_manager.filter_recommendations(recommendations, employee_id)
        result = []
        for rec in filtered:
            if not self.memory_manager.was_similar_title_dismissed(rec.employee_id, rec.title):
                result.append(rec)
        return result

    def get_recommendation_message(self, rec: Recommendation) -> str:
        """Format a recommendation for display in the chat UI."""
        priority_badge = self._priority_badge(rec.priority)
        confidence_bar = self._confidence_bar(rec.confidence)
        lines = [
            f"### {priority_badge} {rec.title}",
            f"\n**Reason:** {rec.reason}",
            f"\n**Business Impact:** {rec.business_impact}",
            f"\n**Suggested Action:** {rec.suggested_action}",
            f"\n**Confidence:** {confidence_bar} ({rec.confidence:.0%})",
        ]
        if rec.approval_required:
            lines.append(f"\n---\n*This requires your approval.* Reply **yes** to approve or **no** to dismiss.")
        else:
            lines.append("\n---\n*This is an informational recommendation.*")
        return "\n".join(lines)

    def handle_user_response(self, response: str, recommendation: Recommendation) -> str | None:
        """Process user's yes/no response to a recommendation."""
        normalized = response.strip().lower()
        if normalized in {"yes", "y", "approve", "approved", "confirm"}:
            self.approval_manager.approve(recommendation.recommendation_id)
            self.memory_manager.record_accepted(recommendation.recommendation_id, recommendation.employee_id, recommendation.title)
            if recommendation.approval_required:
                return self._execute_approved_action(recommendation)
            return f"Recommendation acknowledged: **{recommendation.title}**"
        elif normalized in {"no", "n", "dismiss", "dismissed", "skip"}:
            self.approval_manager.dismiss(recommendation.recommendation_id)
            self.memory_manager.record_dismissed(recommendation.recommendation_id, recommendation.employee_id, recommendation.title)
            return f"Recommendation dismissed: **{recommendation.title}**"
        return None

    def _execute_approved_action(self, rec: Recommendation) -> str:
        """Simulate executing an approved action."""
        self.memory_manager.record_executed(rec.recommendation_id, rec.employee_id, rec.title, rec.suggested_action)
        action_type = rec.approval_type
        if action_type == "send_reminder":
            return f"Reminder sent to manager regarding: **{rec.title}**"
        if action_type == "leave_modification":
            return f"Leave modification request submitted for: **{rec.title}**"
        if action_type == "budget_exception":
            return f"Budget exception request escalated for: **{rec.title}**"
        if action_type == "asset_replacement":
            return f"Asset replacement request submitted for: **{rec.title}**"
        if action_type == "deployment_escalation":
            return f"Deployment conflict escalated for: **{rec.title}**"
        return f"Action executed for: **{rec.title}**"

    def _priority_badge(self, priority: Severity) -> str:
        badges = {
            Severity.CRITICAL: "🔴 CRITICAL",
            Severity.HIGH: "🟠 HIGH",
            Severity.MEDIUM: "🟡 MEDIUM",
            Severity.LOW: "🟢 LOW",
        }
        return badges.get(priority, "⚪ UNKNOWN")

    def _confidence_bar(self, confidence: float) -> str:
        filled = int(confidence * 10)
        return "█" * filled + "░" * (10 - filled)
