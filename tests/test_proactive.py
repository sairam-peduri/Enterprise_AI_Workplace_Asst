"""Tests for the proactive intelligence layer."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.proactive.event_models import (
    EnterpriseEvent,
    CorrelatedEvent,
    Recommendation,
    EventType,
    Severity,
    EventTypeCategory,
    EVENT_TYPE_CATEGORIES,
)
from src.proactive.event_collector import EventCollector
from src.proactive.event_correlator import EventCorrelator
from src.proactive.priority_engine import PriorityEngine
from src.proactive.recommendation_engine import RecommendationEngine
from src.proactive.notification_manager import NotificationManager
from src.proactive.context_manager import ContextManager
from src.proactive.approval_manager import ApprovalManager
from src.proactive.memory_manager import MemoryManager
from src.proactive.proactive_engine import ProactiveEngine


class TestEventModels:
    def test_enterprise_event_creation(self):
        event = EnterpriseEvent(
            employee_id="EMP001",
            employee_name="Priya Sharma",
            department="HR",
            event_type=EventType.TRAINING_EXPIRY,
            severity=Severity.HIGH,
            source_system="LMS",
            title="Security Training Expiring",
            description="Security awareness training expires in 3 days",
            due_date=date.today(),
        )
        assert event.employee_id == "EMP001"
        assert event.severity == Severity.HIGH
        assert event.days_until_due == 0

    def test_enterprise_event_to_dict(self):
        event = EnterpriseEvent(
            event_id="test123",
            employee_id="EMP001",
            employee_name="Priya Sharma",
            department="HR",
            event_type=EventType.TRAINING_EXPIRY,
            severity=Severity.HIGH,
            source_system="LMS",
            title="Security Training Expiring",
            description="Expires soon",
            due_date=date(2026, 8, 1),
        )
        d = event.to_dict()
        assert d["event_id"] == "test123"
        assert d["due_date"] == "2026-08-01"
        assert d["event_type"] == "training_expiry"

    def test_enterprise_event_from_dict(self):
        data = {
            "event_id": "test456",
            "employee_id": "EMP002",
            "employee_name": "Rajesh Patel",
            "department": "Finance",
            "event_type": "expense_blocked",
            "severity": "High",
            "source_system": "Finance System",
            "title": "Expense Blocked",
            "description": "Expense over limit",
            "due_date": "2026-08-15",
            "metadata": {"amount": 5000},
            "timestamp": datetime.now().isoformat(),
        }
        event = EnterpriseEvent.from_dict(data)
        assert event.event_id == "test456"
        assert event.event_type == EventType.EXPENSE_BLOCKED
        assert event.due_date == date(2026, 8, 15)

    def test_correlated_event(self):
        e1 = EnterpriseEvent(employee_id="EMP001", event_type=EventType.TRAINING_EXPIRY, severity=Severity.HIGH, title="Training 1")
        e2 = EnterpriseEvent(employee_id="EMP001", event_type=EventType.CERTIFICATION_EXPIRY, severity=Severity.MEDIUM, title="Cert 1")
        corr = CorrelatedEvent(events=[e1, e2], correlation_reason="Same employee, compliance issues")
        assert len(corr.events) == 2
        assert corr.employee_ids == ["EMP001"]

    def test_recommendation_to_dict(self):
        rec = Recommendation(
            recommendation_id="rec1",
            employee_id="EMP001",
            employee_name="Priya Sharma",
            title="Complete Training",
            reason="Training expires soon",
            business_impact="Compliance risk",
            suggested_action="Enroll in training",
            priority=Severity.HIGH,
            confidence=0.85,
            approval_required=True,
        )
        d = rec.to_dict()
        assert d["recommendation_id"] == "rec1"
        assert d["priority"] == "High"
        assert d["confidence"] == 0.85

    def test_event_type_categories(self):
        assert EVENT_TYPE_CATEGORIES[EventType.TRAINING_EXPIRY] == EventTypeCategory.SECURITY
        assert EVENT_TYPE_CATEGORIES[EventType.EXPENSE_BLOCKED] == EventTypeCategory.FINANCE
        assert EVENT_TYPE_CATEGORIES[EventType.WARRANTY_EXPIRY] == EventTypeCategory.IT


class TestEventCollector:
    def test_collect_all(self):
        collector = EventCollector()
        events = collector.collect_all()
        assert isinstance(events, list)
        assert len(events) > 0
        for event in events:
            assert isinstance(event, EnterpriseEvent)

    def test_collect_by_type(self):
        collector = EventCollector()
        events = collector.collect_by_type(EventType.TRAINING_EXPIRY)
        for event in events:
            assert event.event_type == EventType.TRAINING_EXPIRY

    def test_collect_by_employee(self):
        collector = EventCollector()
        events = collector.collect_for_employee("EMP001")
        for event in events:
            assert event.employee_id == "EMP001"


class TestEventCorrelator:
    def test_correlate(self):
        correlator = EventCorrelator()
        collector = EventCollector()
        events = collector.collect_all()
        correlated = correlator.correlate(events)
        assert isinstance(correlated, list)
        for corr in correlated:
            assert isinstance(corr, CorrelatedEvent)


class TestPriorityEngine:
    def test_score_event(self):
        engine = PriorityEngine()
        event = EnterpriseEvent(
            employee_id="EMP001",
            event_type=EventType.TRAINING_EXPIRY,
            severity=Severity.HIGH,
            title="Test Event",
            due_date=date.today(),
        )
        score = engine.score_event(event)
        assert score.priority in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)
        assert 0.0 <= score.confidence <= 1.0

    def test_score_correlated(self):
        engine = PriorityEngine()
        e1 = EnterpriseEvent(employee_id="EMP001", event_type=EventType.TRAINING_EXPIRY, severity=Severity.HIGH, title="Event 1")
        e2 = EnterpriseEvent(employee_id="EMP001", event_type=EventType.CERTIFICATION_EXPIRY, severity=Severity.MEDIUM, title="Event 2")
        corr = CorrelatedEvent(events=[e1, e2], correlation_reason="Test correlation")
        score = engine.score_correlated(corr)
        assert score.priority in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)


class TestRecommendationEngine:
    def test_generate_batch(self):
        engine = RecommendationEngine()
        correlator = EventCorrelator()
        collector = EventCollector()
        events = collector.collect_all()
        correlated = correlator.correlate(events)
        recs = engine.generate_batch(correlated)
        assert isinstance(recs, list)
        for rec in recs:
            assert isinstance(rec, Recommendation)


class TestNotificationManager:
    def test_filter_recommendations(self):
        manager = NotificationManager()
        rec1 = Recommendation(recommendation_id="r1", employee_id="EMP001", title="Test 1", priority=Severity.HIGH, confidence=0.9)
        rec2 = Recommendation(recommendation_id="r2", employee_id="EMP002", title="Test 2", priority=Severity.LOW, confidence=0.5)
        filtered = manager.filter_recommendations([rec1, rec2], "EMP001")
        assert len(filtered) == 1
        assert filtered[0].employee_id == "EMP001"

    def test_deduplicate(self):
        manager = NotificationManager()
        rec1 = Recommendation(recommendation_id="r1", employee_id="EMP001", title="Training Expiring", priority=Severity.HIGH)
        rec2 = Recommendation(recommendation_id="r2", employee_id="EMP001", title="Training Expiring", priority=Severity.HIGH)
        rec3 = Recommendation(recommendation_id="r3", employee_id="EMP001", title="Different Title", priority=Severity.LOW)
        unique = manager._deduplicate([rec1, rec2, rec3])
        assert len(unique) == 2


class TestContextManager:
    def test_get_employee_context(self):
        manager = ContextManager()
        context = manager.get_employee_context("EMP001")
        assert context["found"] is True
        assert context["name"] == "Priya Sharma"

    def test_get_employee_context_not_found(self):
        manager = ContextManager()
        context = manager.get_employee_context("EMP999")
        assert context["found"] is False


class TestApprovalManager:
    def test_approve_and_dismiss(self):
        manager = ApprovalManager()
        rec = Recommendation(
            recommendation_id="test_rec",
            employee_id="EMP001",
            title="Test",
            reason="Test reason",
            business_impact="Test impact",
            suggested_action="Test action",
            priority=Severity.HIGH,
            approval_required=True,
        )
        manager.request_approval(rec)
        assert manager.is_pending("test_rec") is True
        result = manager.approve("test_rec")
        assert result is True
        assert manager.is_pending("test_rec") is False

        rec2 = Recommendation(
            recommendation_id="test_rec2",
            employee_id="EMP001",
            title="Test2",
            reason="Test reason",
            business_impact="Test impact",
            suggested_action="Test action",
            priority=Severity.LOW,
            approval_required=True,
        )
        manager.request_approval(rec2)
        result2 = manager.dismiss("test_rec2")
        assert result2 is True


class TestMemoryManager:
    def test_record_and_check(self):
        with TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "test_memory.json"
            manager = MemoryManager(mem_file)
            manager.record_dismissed("rec1", "EMP001", "Training Expiring")
            assert manager.is_dismissed("rec1") is True
            assert manager.was_similar_title_dismissed("EMP001", "Training Expiring") is True
            assert manager.was_similar_title_dismissed("EMP001", "Different Title") is False

    def test_stats(self):
        with TemporaryDirectory() as tmpdir:
            mem_file = Path(tmpdir) / "test_memory.json"
            manager = MemoryManager(mem_file)
            manager.record_dismissed("r1", "EMP001", "Test")
            manager.record_accepted("r2", "EMP001", "Test 2")
            stats = manager.stats
            assert stats["dismissed"] == 1
            assert stats["accepted"] == 1


class TestProactiveEngine:
    def test_run_pipeline(self):
        engine = ProactiveEngine()
        recs = engine.run_pipeline()
        assert isinstance(recs, list)
        for rec in recs:
            assert isinstance(rec, Recommendation)

    def test_run_pipeline_filtered(self):
        engine = ProactiveEngine()
        recs = engine.run_pipeline(employee_id="EMP001")
        for rec in recs:
            assert rec.employee_id == "EMP001"

    def test_get_recommendation_message(self):
        engine = ProactiveEngine()
        rec = Recommendation(
            recommendation_id="r1",
            employee_id="EMP001",
            title="Test Recommendation",
            reason="Test reason",
            business_impact="Test impact",
            suggested_action="Test action",
            priority=Severity.HIGH,
            confidence=0.85,
            approval_required=True,
        )
        msg = engine.get_recommendation_message(rec)
        assert "Test Recommendation" in msg
        assert "HIGH" in msg
        assert "approval" in msg.lower()

    def test_handle_user_response_approve(self):
        engine = ProactiveEngine()
        rec = Recommendation(
            recommendation_id="r1",
            employee_id="EMP001",
            title="Test",
            reason="Test",
            business_impact="Test",
            suggested_action="Test",
            priority=Severity.HIGH,
            confidence=0.8,
            approval_required=False,
        )
        result = engine.handle_user_response("yes", rec)
        assert result is not None
        assert "acknowledged" in result.lower()

    def test_handle_user_response_dismiss(self):
        engine = ProactiveEngine()
        rec = Recommendation(
            recommendation_id="r2",
            employee_id="EMP001",
            title="Test Dismiss",
            reason="Test",
            business_impact="Test",
            suggested_action="Test",
            priority=Severity.LOW,
            confidence=0.8,
            approval_required=False,
        )
        result = engine.handle_user_response("no", rec)
        assert result is not None
        assert "dismissed" in result.lower()

    def test_handle_user_response_invalid(self):
        engine = ProactiveEngine()
        rec = Recommendation(recommendation_id="r3", employee_id="EMP001", title="Test")
        result = engine.handle_user_response("maybe", rec)
        assert result is None
