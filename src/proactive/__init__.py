"""Proactive intelligence layer for EnterpriseAssist AI."""

from src.proactive.event_models import EnterpriseEvent, EventType, Severity
from src.proactive.event_collector import EventCollector
from src.proactive.event_correlator import EventCorrelator
from src.proactive.priority_engine import PriorityEngine
from src.proactive.recommendation_engine import RecommendationEngine
from src.proactive.notification_manager import NotificationManager
from src.proactive.context_manager import ContextManager
from src.proactive.approval_manager import ApprovalManager
from src.proactive.memory_manager import MemoryManager

__all__ = [
    "EnterpriseEvent",
    "EventType",
    "Severity",
    "EventCollector",
    "EventCorrelator",
    "PriorityEngine",
    "RecommendationEngine",
    "NotificationManager",
    "ContextManager",
    "ApprovalManager",
    "MemoryManager",
]
