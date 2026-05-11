"""
agent.py
Agentic router: classifies questions and routes to the appropriate tool.

Tools:
  - RAG Tool: For document-based knowledge questions.
  - Appointment Tool: For booking/slot queries (mock).
  - Emergency Tool: For emergency detection (safety-first routing).
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from app.config import MOCK_SLOTS

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    APPOINTMENT = "appointment"
    EMERGENCY = "emergency"
    KNOWLEDGE = "knowledge"


@dataclass
class AgentDecision:
    query_type: QueryType
    tool_used: str
    tool_result: Optional[str]
    route_reason: str


# ── Keyword Patterns ───────────────────────────────────────────────────────────

APPOINTMENT_KEYWORDS = [
    # Explicit booking intent
    r"\bbook\s+(a\s+)?(an\s+)?appointment\b",
    r"\bschedule\s+(a\s+)?(an\s+)?appointment\b",
    r"\bbook\s+(a\s+)?(an\s+)?\w+\s+appointment\b",
    r"\bmake\s+(a\s+)?(an\s+)?appointment\b",
    r"\bget\s+(a\s+)?(an\s+)?appointment\b",
    # Slot availability (with day or department context)
    r"\bany\s+slots?\b",
    r"\bopen\s+slots?\b",
    r"\bavailable\s+slots?\b",
    r"\bcheck\s+slots?\b",
    r"\bslots?\s+(for|on|at)\b",
    r"\bnext\s+available\s+slot\b",
    r"\bappointment\s+(for|on|at|with)\b",
    # Book + department combos
    r"\bbook\b.{0,30}\b(cardiology|orthopedic|dermat|pediatr|endocrinolog|neurol|gynecolog|mental health|general medicine|oncolog)\b",
    r"\bsee\s+a\s+(doctor|specialist|physician)\b",
    r"\bwant\s+to\s+(see|visit)\s+a\s+(doctor|specialist)\b",
]

EMERGENCY_KEYWORDS = [
    r"\bemergency\b", r"\bchest pain\b", r"\bcan't breathe\b", r"\bcannot breathe\b",
    r"\bshortness of breath\b", r"\bunconsciou\b", r"\bseizure\b",
    r"\bstroke\b", r"\bheart attack\b", r"\bsevere bleeding\b",
    r"\boverdose\b", r"\bsuicid\b", r"\bkill myself\b",
]

DEPARTMENTS_PATTERN = [
    "cardiology", "general medicine", "orthopedics", "dermatology",
    "pediatrics", "endocrinology", "mental health", "neurology",
    "gynecology", "ophthalmology", "ent", "oncology",
]

DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "today", "tomorrow", "this week", "next week",
]

DAY_NORMALIZE = {
    "mon": "monday", "tue": "tuesday", "wed": "wednesday",
    "thu": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday",
    "today": "monday",    # mock default
    "tomorrow": "tuesday",
    "this week": "monday",
    "next week": "monday",
}


# ── Router ─────────────────────────────────────────────────────────────────────

def classify_query(question: str) -> QueryType:
    """
    Classify a user question into one of three categories.

    Priority: Emergency > Appointment > Knowledge
    """
    q = question.lower()

    for pattern in EMERGENCY_KEYWORDS:
        if re.search(pattern, q):
            return QueryType.EMERGENCY

    for pattern in APPOINTMENT_KEYWORDS:
        if re.search(pattern, q):
            return QueryType.APPOINTMENT

    return QueryType.KNOWLEDGE


# ── Tools ──────────────────────────────────────────────────────────────────────

def check_available_slots(department: str, day: str) -> str:
    """
    Mock appointment slot checker tool.

    Args:
        department: Department name.
        day: Day of the week.

    Returns:
        Human-readable availability string.
    """
    dept_key = department.lower().strip()
    day_key = DAY_NORMALIZE.get(day.lower().strip(), day.lower().strip())

    # Find closest matching department
    matched_dept = None
    for dept in MOCK_SLOTS:
        if dept_key in dept or dept in dept_key:
            matched_dept = dept
            break

    if matched_dept is None:
        available_depts = ", ".join(d.title() for d in MOCK_SLOTS.keys())
        return (
            f"I don't have slot information for the '{department}' department. "
            f"Departments with available telehealth/outpatient slots include: {available_depts}. "
            "You can also call 1800-MED-APPT (1800-633-2778) or book via the patient portal."
        )

    dept_schedule = MOCK_SLOTS[matched_dept]

    if day_key not in dept_schedule:
        available_days = ", ".join(d.title() for d in dept_schedule.keys())
        return (
            f"The {matched_dept.title()} department is not available on {day.title()}. "
            f"Available days are: {available_days}. "
            "Would you like to check slots on one of those days?"
        )

    slots = dept_schedule[day_key]
    slots_str = ", ".join(slots)
    return (
        f"✅ Available slots for **{matched_dept.title()}** on **{day_key.title()}**:\n"
        f"{slots_str}\n\n"
        "To confirm a booking, please visit the MediCare Patient Portal at portal.medicare-hs.com "
        "or call 1800-MED-APPT (1800-633-2778)."
    )


def extract_appointment_details(question: str) -> Tuple[str, str]:
    """
    Extract department and day from a natural language appointment question.

    Returns:
        Tuple of (department, day) — defaults to 'general medicine' and 'monday' if not found.
    """
    q = question.lower()

    detected_dept = "general medicine"
    for dept in DEPARTMENTS_PATTERN:
        if dept in q:
            detected_dept = dept
            break

    # Check for partial match / natural language aliases
    dept_aliases = {
        "cardio": "cardiology", "cardiologist": "cardiology",
        "heart specialist": "cardiology", "heart doctor": "cardiology", "heart": "cardiology",
        "ortho": "orthopedics", "bone": "orthopedics", "joint": "orthopedics",
        "skin": "dermatology", "derm": "dermatology",
        "child": "pediatrics", "kid": "pediatrics", "baby": "pediatrics", "infant": "pediatrics",
        "diabetes": "endocrinology", "thyroid": "endocrinology", "hormone": "endocrinology",
        "mental": "mental health", "therapy": "mental health", "counseling": "mental health", "psychiatr": "mental health",
        "brain": "neurology", "neuro": "neurology", "neurologist": "neurology",
        "women": "gynecology", "gyno": "gynecology", "gynecologist": "gynecology",
        "eye": "ophthalmology", "vision": "ophthalmology",
        "ear": "ent", "nose": "ent", "throat": "ent",
        "cancer": "oncology", "oncologist": "oncology",
    }
    # Check multi-word aliases first (longer matches take priority)
    for alias in sorted(dept_aliases, key=len, reverse=True):
        if alias in q and detected_dept == "general medicine":
            detected_dept = dept_aliases[alias]
            break

    # Detect day - check multi-word phrases first
    detected_day = "monday"
    day_priority = ["this week", "next week", "tomorrow", "today",
                    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                    "mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    for day in day_priority:
        if day in q:
            detected_day = DAY_NORMALIZE.get(day, day)
            break

    return detected_dept, detected_day


def handle_emergency(question: str) -> str:
    """Return immediate emergency guidance."""
    return (
        "⚠️ This sounds like a potential medical emergency.\n\n"
        "**Please call 112 immediately or go to your nearest Emergency Room.**\n\n"
        "Do not delay seeking emergency care. MediCare Health System's Emergency Department "
        "is available 24/7. If you are at our facility, please proceed to the Emergency entrance "
        "immediately and inform the staff."
    )


# ── Main Agent Entry Point ─────────────────────────────────────────────────────

def route_query(question: str) -> AgentDecision:
    """
    Route a question to the appropriate tool and return an AgentDecision.
    For KNOWLEDGE type, returns None as tool_result (handled by RAG + LLM separately).

    Args:
        question: The user's question.

    Returns:
        AgentDecision with routing info and tool result (if applicable).
    """
    query_type = classify_query(question)
    logger.info(f"Query classified as: {query_type.value} — '{question[:80]}'")

    if query_type == QueryType.EMERGENCY:
        return AgentDecision(
            query_type=query_type,
            tool_used="emergency_handler",
            tool_result=handle_emergency(question),
            route_reason="Emergency keywords detected — routed to emergency response tool.",
        )

    if query_type == QueryType.APPOINTMENT:
        department, day = extract_appointment_details(question)
        slot_result = check_available_slots(department, day)
        return AgentDecision(
            query_type=query_type,
            tool_used="check_available_slots",
            tool_result=slot_result,
            route_reason=f"Appointment intent detected — checked slots for {department} on {day}.",
        )

    # KNOWLEDGE: let the RAG + LLM pipeline handle it
    return AgentDecision(
        query_type=query_type,
        tool_used="rag_pipeline",
        tool_result=None,
        route_reason="General knowledge question — routed to RAG pipeline.",
    )