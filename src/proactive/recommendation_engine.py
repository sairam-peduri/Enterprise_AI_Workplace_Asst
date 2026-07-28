"""Recommendation engine that converts correlated events into actionable recommendations."""

from __future__ import annotations

from datetime import date

from src.proactive.event_models import (
    CorrelatedEvent,
    EnterpriseEvent,
    EventType,
    Recommendation,
    Severity,
)
from src.proactive.priority_engine import PriorityEngine


# Template-based recommendation generation
RECOMMENDATION_TEMPLATES: dict[EventType, dict] = {
    EventType.TRAINING_EXPIRY: {
        "title_template": "Complete {course_name} Before Deadline",
        "reason_template": "Your mandatory training '{course_name}' expires on {due_date}. Non-compliance may affect your performance review and departmental compliance metrics.",
        "impact_template": "Failure to complete may result in access restrictions and compliance violations.",
        "action_template": "Complete the training module before the deadline.",
        "approval_required": False,
    },
    EventType.PASSPORT_EXPIRY: {
        "title_template": "Renew Passport Before Travel",
        "reason_template": "Your passport expires on {due_date} ({days_remaining} days). Most international destinations require at least 6 months validity. Your upcoming travel to {destination} may be affected.",
        "impact_template": "International travel approval may become invalid. Flight and hotel bookings may need cancellation.",
        "action_template": "Initiate passport renewal immediately. Contact HR for assistance with expedited processing.",
        "approval_required": False,
    },
    EventType.EXPENSE_BLOCKED: {
        "title_template": "Upload Missing Document for {expense_id}",
        "reason_template": "Your expense claim {expense_id} for ₹{amount} is blocked due to a missing {missing_document}.",
        "impact_template": "Reimbursement will be delayed until the document is uploaded.",
        "action_template": "Upload the required {missing_document} to unblock your expense claim.",
        "approval_required": False,
    },
    EventType.LEAVE_CONFLICT: {
        "title_template": "Resolve Leave and Deployment Conflict",
        "reason_template": "Your approved leave overlaps with a scheduled production deployment. Coordination with your manager is recommended.",
        "impact_template": "Production deployment may be impacted if you are unavailable.",
        "action_template": "Discuss with your manager to either adjust your leave or delegate deployment responsibilities.",
        "approval_required": True,
        "approval_type": "leave_modification",
    },
    EventType.WARRANTY_EXPIRY: {
        "title_template": "Plan {device} Replacement",
        "reason_template": "Your {device} warranty expires on {due_date} ({days_remaining} days). Without warranty, repairs will be at personal cost.",
        "impact_template": "Hardware failure after warranty expiry may cause productivity loss.",
        "action_template": "Discuss replacement options with IT or request warranty extension.",
        "approval_required": True,
        "approval_type": "asset_replacement",
    },
    EventType.BUDGET_EXCEEDED: {
        "title_template": "Request Budget Exception for Travel",
        "reason_template": "Your travel request to {destination} exceeds the departmental budget of ₹{budget_limit} by ₹{excess_amount}.",
        "impact_template": "Travel may not proceed without budget exception approval.",
        "action_template": "Request a budget exception from your department head.",
        "approval_required": True,
        "approval_type": "budget_exception",
    },
    EventType.PENDING_APPROVAL: {
        "title_template": "Follow Up on {title}",
        "reason_template": "Your {request_type} has been pending approval for {days_pending} working days.",
        "impact_template": "Delays may affect travel plans or reimbursement timelines.",
        "action_template": "Send a reminder to your manager or escalate if urgent.",
        "approval_required": True,
        "approval_type": "send_reminder",
    },
    EventType.DEPLOYMENT_CONFLICT: {
        "title_template": "Coordinate Deployment and Leave",
        "reason_template": "A production deployment is scheduled during your approved leave period.",
        "impact_template": "Production stability may be at risk without proper handover.",
        "action_template": "Delegate deployment responsibilities or adjust leave dates.",
        "approval_required": True,
        "approval_type": "deployment_escalation",
    },
}


class RecommendationEngine:
    """Converts correlated events and priority scores into recommendations."""

    def __init__(self):
        self.priority_engine = PriorityEngine()

    def generate_from_correlated(self, correlated: CorrelatedEvent) -> Recommendation:
        """Generate a recommendation from a correlated event group."""
        primary_event = correlated.events[0]
        score = self.priority_engine.score_correlated(correlated)
        template = RECOMMENDATION_TEMPLATES.get(primary_event.event_type, self._default_template(primary_event))
        title = self._fill_template(template["title_template"], primary_event, correlated)
        reason = self._fill_template(template["reason_template"], primary_event, correlated)
        impact = self._fill_template(template["impact_template"], primary_event, correlated)
        action = self._fill_template(template["action_template"], primary_event, correlated)
        return Recommendation(
            employee_id=primary_event.employee_id,
            employee_name=primary_event.employee_name,
            title=title,
            reason=reason,
            business_impact=impact,
            suggested_action=action,
            priority=score.priority,
            confidence=score.confidence,
            approval_required=template.get("approval_required", False),
            approval_type=template.get("approval_type", ""),
            correlated_events=correlated.events,
        )

    def generate_from_single(self, event: EnterpriseEvent) -> Recommendation:
        """Generate a recommendation from a single event."""
        correlated = CorrelatedEvent(events=[event], correlation_reason=event.description)
        return self.generate_from_correlated(correlated)

    def generate_batch(self, correlated_events: list[CorrelatedEvent]) -> list[Recommendation]:
        """Generate recommendations for a batch of correlated events."""
        recs = [self.generate_from_correlated(c) for c in correlated_events]
        recs.sort(key=lambda r: (r.priority.value, -r.confidence), reverse=True)
        return recs

    def _fill_template(self, template: str, event: EnterpriseEvent, correlated: CorrelatedEvent) -> str:
        days_remaining = event.days_until_due or "unknown"
        meta = {**event.metadata}
        meta.setdefault("due_date", str(event.due_date) if event.due_date else "unknown")
        meta.setdefault("days_remaining", str(days_remaining))
        meta.setdefault("course_name", event.title)
        meta.setdefault("destination", meta.get("destination", "overseas"))
        meta.setdefault("expense_id", meta.get("expense_id", "N/A"))
        meta.setdefault("amount", str(meta.get("amount", 0)))
        meta.setdefault("missing_document", meta.get("missing_document", "document"))
        meta.setdefault("device", meta.get("device", "device"))
        meta.setdefault("budget_limit", str(meta.get("budget_limit", 0)))
        meta.setdefault("title", event.title)
        meta.setdefault("request_type", event.title)
        try:
            return template.format(**meta)
        except (KeyError, IndexError):
            return template

    def _default_template(self, event: EnterpriseEvent) -> dict:
        return {
            "title_template": event.title,
            "reason_template": event.description,
            "impact_template": "Requires attention to avoid disruption.",
            "action_template": "Review and take appropriate action.",
            "approval_required": False,
        }
