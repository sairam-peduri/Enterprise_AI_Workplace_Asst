"""Enterprise event models for the proactive intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    """Types of enterprise events."""

    TRAINING_EXPIRY = "training_expiry"
    PASSPORT_EXPIRY = "passport_expiry"
    EXPENSE_BLOCKED = "expense_blocked"
    LEAVE_CONFLICT = "leave_conflict"
    WARRANTY_EXPIRY = "warranty_expiry"
    BUDGET_EXCEEDED = "budget_exceeded"
    PENDING_APPROVAL = "pending_approval"
    DEPLOYMENT_CONFLICT = "deployment_conflict"
    VISA_EXPIRY = "visa_expiry"
    CERTIFICATION_EXPIRY = "certification_expiry"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    POLICY_VIOLATION = "policy_violation"


class Severity(str, Enum):
    """Event severity levels."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EventTypeCategory(str, Enum):
    """Broader categories for event types."""

    SECURITY = "security"
    COMPLIANCE = "compliance"
    FINANCE = "finance"
    OPERATIONS = "operations"
    HR = "hr"
    IT = "it"
    TRAVEL = "travel"


EVENT_TYPE_CATEGORIES: dict[EventType, EventTypeCategory] = {
    EventType.TRAINING_EXPIRY: EventTypeCategory.SECURITY,
    EventType.PASSPORT_EXPIRY: EventTypeCategory.TRAVEL,
    EventType.EXPENSE_BLOCKED: EventTypeCategory.FINANCE,
    EventType.LEAVE_CONFLICT: EventTypeCategory.HR,
    EventType.WARRANTY_EXPIRY: EventTypeCategory.IT,
    EventType.BUDGET_EXCEEDED: EventTypeCategory.FINANCE,
    EventType.PENDING_APPROVAL: EventTypeCategory.OPERATIONS,
    EventType.DEPLOYMENT_CONFLICT: EventTypeCategory.OPERATIONS,
    EventType.VISA_EXPIRY: EventTypeCategory.TRAVEL,
    EventType.CERTIFICATION_EXPIRY: EventTypeCategory.COMPLIANCE,
    EventType.EQUIPMENT_MALFUNCTION: EventTypeCategory.IT,
    EventType.POLICY_VIOLATION: EventTypeCategory.COMPLIANCE,
}


@dataclass
class EnterpriseEvent:
    """Represents a single enterprise event requiring attention."""

    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    employee_id: str = ""
    employee_name: str = ""
    department: str = ""
    event_type: EventType = EventType.TRAINING_EXPIRY
    severity: Severity = Severity.MEDIUM
    source_system: str = ""
    title: str = ""
    description: str = ""
    due_date: date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "department": self.department,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source_system": self.source_system,
            "title": self.title,
            "description": self.description,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnterpriseEvent:
        due = data.get("due_date")
        if due and isinstance(due, str):
            due = date.fromisoformat(due)
        ts = data.get("timestamp")
        if ts and isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            event_id=data.get("event_id", str(uuid4())[:8]),
            employee_id=data.get("employee_id", ""),
            employee_name=data.get("employee_name", ""),
            department=data.get("department", ""),
            event_type=EventType(data.get("event_type", "training_expiry")),
            severity=Severity(data.get("severity", "Medium")),
            source_system=data.get("source_system", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            due_date=due,
            metadata=data.get("metadata", {}),
            timestamp=ts if isinstance(ts, datetime) else datetime.now(),
        )

    @property
    def days_until_due(self) -> int | None:
        if self.due_date is None:
            return None
        return (self.due_date - date.today()).days

    @property
    def category(self) -> EventTypeCategory:
        return EVENT_TYPE_CATEGORIES.get(self.event_type, EventTypeCategory.OPERATIONS)


@dataclass
class CorrelatedEvent:
    """A group of related events correlated together."""

    correlation_id: str = field(default_factory=lambda: str(uuid4())[:8])
    events: list[EnterpriseEvent] = field(default_factory=list)
    correlation_reason: str = ""
    combined_severity: Severity = Severity.MEDIUM

    @property
    def employee_ids(self) -> list[str]:
        return list({e.employee_id for e in self.events})

    @property
    def event_types(self) -> list[EventType]:
        return list({e.event_type for e in self.events})

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "events": [e.to_dict() for e in self.events],
            "correlation_reason": self.correlation_reason,
            "combined_severity": self.combined_severity.value,
        }


@dataclass
class Recommendation:
    """A proactive recommendation for an employee."""

    recommendation_id: str = field(default_factory=lambda: str(uuid4())[:8])
    employee_id: str = ""
    employee_name: str = ""
    title: str = ""
    reason: str = ""
    business_impact: str = ""
    suggested_action: str = ""
    priority: Severity = Severity.MEDIUM
    confidence: float = 0.8
    approval_required: bool = False
    approval_type: str = ""
    correlated_events: list[EnterpriseEvent] = field(default_factory=list)
    dismissed: bool = False
    accepted: bool = False
    executed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "title": self.title,
            "reason": self.reason,
            "business_impact": self.business_impact,
            "suggested_action": self.suggested_action,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "approval_required": self.approval_required,
            "approval_type": self.approval_type,
            "events": [e.to_dict() for e in self.correlated_events],
            "dismissed": self.dismissed,
            "accepted": self.accepted,
            "executed": self.executed,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
