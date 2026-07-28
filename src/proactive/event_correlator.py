"""Event correlator that identifies related enterprise events."""

from __future__ import annotations

from datetime import date, timedelta

from src.proactive.event_models import (
    CorrelatedEvent,
    EnterpriseEvent,
    EventType,
    Severity,
)

# Type aliases
CorrelationRule = tuple[str, list[EventType], callable]


def _travel_passport_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    travel = [e for e in events if e.event_type == EventType.PENDING_APPROVAL and e.metadata.get("destination")]
    passport = [e for e in events if e.event_type == EventType.PASSPORT_EXPIRY]
    results = []
    for t in travel:
        for p in passport:
            if t.employee_id == p.employee_id and p.due_date and p.days_until_due is not None and p.days_until_due < 180:
                results.append(CorrelatedEvent(
                    events=[t, p],
                    correlation_reason=f"Travel to {t.metadata.get('destination', 'overseas')} approved but passport expires in {p.days_until_due} days. Most destinations require 6 months validity.",
                    combined_severity=Severity.CRITICAL,
                ))
    return results


def _leave_deployment_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    leave = [e for e in events if e.event_type == EventType.LEAVE_CONFLICT]
    deploy = [e for e in events if e.event_type == EventType.DEPLOYMENT_CONFLICT]
    results = []
    for l in leave:
        for d in deploy:
            if l.employee_id == d.employee_id:
                results.append(CorrelatedEvent(
                    events=[l, d],
                    correlation_reason=f"Approved leave overlaps with production deployment {d.metadata.get('deployment_id', '')}. Coordination needed.",
                    combined_severity=Severity.HIGH,
                ))
    return results


def _expense_missing_doc_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    expenses = [e for e in events if e.event_type == EventType.EXPENSE_BLOCKED]
    results = []
    for exp in expenses:
        results.append(CorrelatedEvent(
            events=[exp],
            correlation_reason=f"Expense {exp.metadata.get('expense_id', '')} blocked: missing {exp.metadata.get('missing_document', 'document')}.",
            combined_severity=exp.severity,
        ))
    return results


def _warranty_approaching_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    warranties = [e for e in events if e.event_type == EventType.WARRANTY_EXPIRY and e.days_until_due is not None and e.days_until_due <= 14]
    results = []
    for w in warranties:
        results.append(CorrelatedEvent(
            events=[w],
            correlation_reason=f"{w.metadata.get('device', 'Device')} warranty expires in {w.days_until_due} days. Plan replacement or renewal.",
            combined_severity=Severity.MEDIUM,
        ))
    return results


def _pending_approval_stale_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    approvals = [e for e in events if e.event_type == EventType.PENDING_APPROVAL and e.metadata.get("days_pending", 0) >= 5]
    results = []
    for a in approvals:
        results.append(CorrelatedEvent(
            events=[a],
            correlation_reason=f"{a.title} has been pending for {a.metadata.get('days_pending', 0)} working days. Consider sending a reminder.",
            combined_severity=Severity.MEDIUM,
        ))
    return results


def _training_expiry_rule(events: list[EnterpriseEvent]) -> CorrelatedEvent | None:
    training = [e for e in events if e.event_type == EventType.TRAINING_EXPIRY and e.days_until_due is not None and e.days_until_due <= 7]
    results = []
    for t in training:
        sev = Severity.CRITICAL if t.days_until_due <= 1 else Severity.HIGH
        results.append(CorrelatedEvent(
            events=[t],
            correlation_reason=f"Mandatory training '{t.title}' expires in {t.days_until_due} day(s). Non-compliance may affect performance review.",
            combined_severity=sev,
        ))
    return results


DEFAULT_RULES: list[callable] = [
    _travel_passport_rule,
    _leave_deployment_rule,
    _expense_missing_doc_rule,
    _warranty_approaching_rule,
    _pending_approval_stale_rule,
    _training_expiry_rule,
]


class EventCorrelator:
    """Correlates enterprise events using configurable rules."""

    def __init__(self, rules: list[callable] | None = None):
        self.rules = rules or DEFAULT_RULES

    def correlate(self, events: list[EnterpriseEvent]) -> list[CorrelatedEvent]:
        """Run all correlation rules and return grouped events."""
        correlated: list[CorrelatedEvent] = []
        for rule in self.rules:
            result = rule(events)
            if result is not None:
                if isinstance(result, list):
                    correlated.extend(result)
                else:
                    correlated.append(result)
        return correlated

    def correlate_for_employee(self, events: list[EnterpriseEvent], employee_id: str) -> list[CorrelatedEvent]:
        """Return correlations relevant to a specific employee."""
        emp_events = [e for e in events if e.employee_id == employee_id]
        return self.correlate(emp_events)
