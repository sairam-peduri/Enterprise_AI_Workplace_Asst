"""Priority engine that scores and ranks enterprise events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.proactive.event_models import (
    CorrelatedEvent,
    EnterpriseEvent,
    EventType,
    Severity,
)


@dataclass
class PriorityScore:
    """Priority assessment for an event or correlation."""

    priority: Severity
    confidence: float
    urgency_score: float
    impact_score: float
    reasoning: str


# Weight factors
URGENCY_WEIGHT = 0.35
IMPACT_WEIGHT = 0.35
CONTEXT_WEIGHT = 0.30

# Severity base scores
SEVERITY_BASE: dict[Severity, float] = {
    Severity.CRITICAL: 0.95,
    Severity.HIGH: 0.75,
    Severity.MEDIUM: 0.50,
    Severity.LOW: 0.25,
}

# Event type impact scores
EVENT_IMPACT: dict[EventType, float] = {
    EventType.TRAINING_EXPIRY: 0.85,
    EventType.PASSPORT_EXPIRY: 0.90,
    EventType.EXPENSE_BLOCKED: 0.50,
    EventType.LEAVE_CONFLICT: 0.80,
    EventType.WARRANTY_EXPIRY: 0.45,
    EventType.BUDGET_EXCEEDED: 0.70,
    EventType.PENDING_APPROVAL: 0.60,
    EventType.DEPLOYMENT_CONFLICT: 0.90,
    EventType.VISA_EXPIRY: 0.85,
    EventType.CERTIFICATION_EXPIRY: 0.80,
    EventType.EQUIPMENT_MALFUNCTION: 0.65,
    EventType.POLICY_VIOLATION: 0.75,
}

# Department criticality
DEPT_CRITICALITY: dict[str, float] = {
    "IT": 0.85,
    "Finance": 0.75,
    "HR": 0.70,
    "Sales": 0.65,
    "Operations": 0.70,
}


class PriorityEngine:
    """Assigns priority scores to events based on urgency, impact, and context."""

    def score_event(self, event: EnterpriseEvent) -> PriorityScore:
        """Score a single enterprise event."""
        urgency = self._compute_urgency(event)
        impact = self._compute_impact(event)
        context = self._compute_context(event)
        raw = (URGENCY_WEIGHT * urgency) + (IMPACT_WEIGHT * impact) + (CONTEXT_WEIGHT * context)
        confidence = min(0.99, max(0.1, raw))
        priority = self._raw_to_priority(raw)
        reasoning = self._build_reasoning(event, urgency, impact, context)
        return PriorityScore(priority=priority, confidence=confidence, urgency_score=urgency, impact_score=impact, reasoning=reasoning)

    def score_correlated(self, correlated: CorrelatedEvent) -> PriorityScore:
        """Score a correlated event group."""
        scores = [self.score_event(e) for e in correlated.events]
        max_raw = max(((URGENCY_WEIGHT * s.urgency_score) + (IMPACT_WEIGHT * s.impact_score) + (CONTEXT_WEIGHT * 0.5)) for s in scores)
        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        confidence = min(0.99, max(0.1, (max_raw + avg_confidence) / 2))
        priority = self._raw_to_priority(max_raw)
        reasoning = f"Combined assessment of {len(correlated.events)} related events. {correlated.correlation_reason}"
        return PriorityScore(priority=priority, confidence=confidence, urgency_score=max_raw, impact_score=max(s.impact_score for s in scores), reasoning=reasoning)

    def _compute_urgency(self, event: EnterpriseEvent) -> float:
        days = event.days_until_due
        if days is None:
            return 0.5
        if days <= 0:
            return 1.0
        if days <= 3:
            return 0.95
        if days <= 7:
            return 0.80
        if days <= 14:
            return 0.65
        if days <= 30:
            return 0.50
        return 0.30

    def _compute_impact(self, event: EnterpriseEvent) -> float:
        base = EVENT_IMPACT.get(event.event_type, 0.50)
        dept = DEPT_CRITICALITY.get(event.department, 0.50)
        severity_factor = SEVERITY_BASE.get(event.severity, 0.50)
        financial = 0.0
        amount = event.metadata.get("amount", 0) or event.metadata.get("estimated_cost", 0)
        if amount:
            financial = min(1.0, amount / 200000)
        return min(1.0, (base * 0.4 + severity_factor * 0.3 + dept * 0.2 + financial * 0.1))

    def _compute_context(self, event: EnterpriseEvent) -> float:
        score = 0.5
        if event.metadata.get("travel_planned"):
            score += 0.2
        if event.metadata.get("days_pending", 0) >= 7:
            score += 0.15
        if event.severity == Severity.CRITICAL:
            score += 0.15
        return min(1.0, score)

    def _raw_to_priority(self, raw: float) -> Severity:
        if raw >= 0.85:
            return Severity.CRITICAL
        if raw >= 0.65:
            return Severity.HIGH
        if raw >= 0.40:
            return Severity.MEDIUM
        return Severity.LOW

    def _build_reasoning(self, event: EnterpriseEvent, urgency: float, impact: float, context: float) -> str:
        parts = [f"Event: {event.title}"]
        days = event.days_until_due
        if days is not None:
            parts.append(f"Due in {days} day(s)")
        parts.append(f"Urgency={urgency:.0%}, Impact={impact:.0%}, Context={context:.0%}")
        return ". ".join(parts)
