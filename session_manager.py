"""
session_manager.py  (v2 — extended for Features 1, 2, 3)
----------------------------------------------------------
Persistence layer for all session data.

NEW in v2:
  • QuestionRecord  — full per-question log (question, options, user pick, correct, feedback)
  • TopicCategory   — three-tier classification (strong / weak / very_weak)
  • StudySheet      — auto-generated study material (definitions, keywords, formulas, notes)
  • SessionSummary  — extended with question_log, topic_categories, study_sheet, incorrect_count
  • SessionState    — answers list now stores full QuestionRecord-compatible dicts
  • build_session_summary — populates all new fields
  • save_study_sheet / load_study_sheet — separate file per session
  • load_session_detail — loads summary + study sheet for History Viewer
  • Backward-compat: all new fields use .get() with safe defaults when reading old JSON files
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
from difficulty_engine import DifficultyState

SESSIONS_DIR = Path.home() / ".study_assistant" / "sessions"


# ─────────────────────────────────────────────────────────────────────────────
# NEW DATACLASS: QuestionRecord
# Stores the complete log of one question attempt for the History Viewer.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuestionRecord:
    """Full log of a single question attempt — used by History Viewer."""
    question_num: int
    question_text: str
    options: List[str]          # all 4 options as displayed (without A)/B)/etc prefix)
    correct_index: int          # 0-based
    user_index: int             # 0-based
    is_correct: bool
    topic: str
    difficulty: str
    question_type: str
    response_time: float
    explanation: str            # feedback explanation shown to user
    reinforcement: str          # memory tip shown
    is_followup: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# NEW DATACLASS: TopicCategory
# Three-tier classification for Feature 2 analytics.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TopicCategory:
    """Classifies topics into strong / weak / very_weak tiers."""
    strong: List[str] = field(default_factory=list)      # accuracy >= 80%
    weak: List[str] = field(default_factory=list)         # accuracy 50–79%
    very_weak: List[str] = field(default_factory=list)    # accuracy < 50%

    # Per-topic detail: topic → {attempts, correct, accuracy}
    detail: Dict[str, dict] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# NEW DATACLASS: StudySheet
# Auto-generated study material for Feature 3.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StudySheet:
    """Study material generated from a completed session."""
    session_id: str
    pdf_title: str
    generated_at: str           # ISO datetime string

    definitions: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"term": "...", "definition": "..."}

    keywords: List[str] = field(default_factory=list)
    # Important terms and exam-relevant vocabulary

    formulas: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"name": "...", "formula": "...", "notes": "..."}

    revision_notes: List[str] = field(default_factory=list)
    # Concise bullet-point revision notes


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED: SessionSummary
# All original fields preserved; new fields added at the end with defaults
# so old JSON files still deserialise cleanly via _load_summary_safe().
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SessionSummary:
    """Snapshot stored at end of session. Extended in v2."""
    # ── Original fields (unchanged) ──────────────────────────────────────────
    session_id: str
    pdf_path: str
    pdf_title: str
    date: str
    total_questions: int
    correct_answers: int
    accuracy_percent: float
    avg_response_time: float
    difficulty_used: str
    mode: str
    topic_accuracy: Dict[str, float]
    weak_topics: List[str]
    strong_topics: List[str]
    difficulty_history: List[str]
    duration_minutes: float

    # ── NEW v2 fields ─────────────────────────────────────────────────────────
    incorrect_answers: int = 0

    # Full per-question log for History Viewer (Feature 1)
    question_log: List[dict] = field(default_factory=list)

    # Three-tier topic categories (Feature 2)
    topic_categories: dict = field(default_factory=dict)
    # Stored as dict (asdict of TopicCategory) for JSON compatibility

    # Flag to tell UI whether a study sheet exists
    has_study_sheet: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED: SessionState
# answers list now stores full QuestionRecord-compatible dicts.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SessionState:
    """Full session state for mid-session save/resume."""
    session_id: str
    pdf_path: str
    pdf_title: str
    start_time: float
    difficulty: str
    mode: str
    question_index: int = 0
    # Each element is a QuestionRecord-compatible dict (superset of original)
    answers: List[dict] = field(default_factory=list)
    topic_correct: Dict[str, int] = field(default_factory=dict)
    topic_total: Dict[str, int] = field(default_factory=dict)
    is_complete: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ID GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_session_id() -> str:
    """Generate a unique 8-char session ID."""
    import hashlib
    ts = str(time.time())
    return hashlib.md5(ts.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE — progress (mid-session)
# ─────────────────────────────────────────────────────────────────────────────
def save_session_progress(state: SessionState) -> str:
    """Save mid-session state to disk. Returns file path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SESSIONS_DIR / f"session_{state.session_id}_progress.json"
    with open(filepath, "w") as f:
        json.dump(asdict(state), f, indent=2)
    return str(filepath)


def load_session_progress(session_id: str) -> Optional[SessionState]:
    """Load a mid-session state by ID."""
    filepath = SESSIONS_DIR / f"session_{session_id}_progress.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return SessionState(**data)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE — summary
# ─────────────────────────────────────────────────────────────────────────────
def save_session_summary(summary: SessionSummary) -> str:
    """Save completed session summary to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SESSIONS_DIR / f"session_{summary.session_id}_summary.json"
    with open(filepath, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    return str(filepath)


def _load_summary_safe(filepath: Path) -> Optional[dict]:
    """
    Load a summary JSON with backward compatibility.
    New v2 fields default to safe empty values if missing from old files.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
        # Inject defaults for any v2 field an older file might be missing
        data.setdefault("incorrect_answers", data.get("total_questions", 0) - data.get("correct_answers", 0))
        data.setdefault("question_log", [])
        data.setdefault("topic_categories", {})
        data.setdefault("has_study_sheet", False)
        return data
    except Exception:
        return None


def list_saved_sessions() -> List[dict]:
    """Return list of all saved session summaries (newest first)."""
    if not SESSIONS_DIR.exists():
        return []
    sessions = []
    for filepath in sorted(SESSIONS_DIR.glob("*_summary.json"), reverse=True):
        data = _load_summary_safe(filepath)
        if data:
            sessions.append(data)
    return sessions


def load_session_detail(session_id: str) -> Optional[dict]:
    """
    Load full session data for the History Viewer.
    Returns summary dict merged with study_sheet if it exists.
    """
    filepath = SESSIONS_DIR / f"session_{session_id}_summary.json"
    data = _load_summary_safe(filepath)
    if not data:
        return None
    # Attach study sheet if present
    sheet = load_study_sheet(session_id)
    if sheet:
        data["study_sheet"] = asdict(sheet)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE — study sheet
# ─────────────────────────────────────────────────────────────────────────────
def save_study_sheet(sheet: StudySheet) -> str:
    """Save study sheet to a separate file next to the summary."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SESSIONS_DIR / f"session_{sheet.session_id}_sheet.json"
    with open(filepath, "w") as f:
        json.dump(asdict(sheet), f, indent=2)
    return str(filepath)


def load_study_sheet(session_id: str) -> Optional[StudySheet]:
    """Load study sheet for a session. Returns None if not present."""
    filepath = SESSIONS_DIR / f"session_{session_id}_sheet.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return StudySheet(**data)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INCOMPLETE SESSION LIST
# ─────────────────────────────────────────────────────────────────────────────
def list_incomplete_sessions() -> List[dict]:
    """Return sessions that were saved mid-way."""
    if not SESSIONS_DIR.exists():
        return []
    sessions = []
    for filepath in sorted(SESSIONS_DIR.glob("*_progress.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
            if not data.get("is_complete", False):
                sessions.append(data)
        except Exception:
            continue
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC CATEGORY BUILDER  (Feature 2)
# ─────────────────────────────────────────────────────────────────────────────
def build_topic_categories(
    topic_accuracy: Dict[str, float],
    topic_correct: Dict[str, int],
    topic_total: Dict[str, int],
) -> TopicCategory:
    """
    Classify topics into three tiers based on accuracy.
      strong    >= 80%
      weak       50–79%
      very_weak < 50%
    Only includes topics with at least 1 attempt.
    """
    cat = TopicCategory()
    for topic, acc in topic_accuracy.items():
        total = topic_total.get(topic, 0)
        correct = topic_correct.get(topic, 0)
        if total == 0:
            continue
        detail_entry = {
            "attempts": total,
            "correct": correct,
            "accuracy": round(acc, 1),
        }
        cat.detail[topic] = detail_entry
        if acc >= 80.0:
            cat.strong.append(topic)
        elif acc >= 50.0:
            cat.weak.append(topic)
        else:
            cat.very_weak.append(topic)
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY BUILDER  (extended)
# ─────────────────────────────────────────────────────────────────────────────
def build_session_summary(
    state: SessionState,
    difficulty_state: DifficultyState,
    end_time: float,
) -> SessionSummary:
    """
    Build a complete SessionSummary from a finished session.
    Populates all v2 fields: question_log, topic_categories, incorrect_answers.
    """
    from datetime import datetime

    duration = (end_time - state.start_time) / 60
    correct = sum(1 for r in difficulty_state.history if r.was_correct)
    total = len(difficulty_state.history)
    topic_acc = difficulty_state.get_topic_accuracy_map()

    # Build three-tier topic classification
    topic_cats = build_topic_categories(
        topic_accuracy=topic_acc,
        topic_correct=difficulty_state.topic_correct,
        topic_total=difficulty_state.topic_total,
    )

    # question_log comes from state.answers (populated in server.py / quiz_engine.py)
    question_log = state.answers  # already full QuestionRecord-compatible dicts

    return SessionSummary(
        session_id=state.session_id,
        pdf_path=state.pdf_path,
        pdf_title=state.pdf_title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_questions=total,
        correct_answers=correct,
        incorrect_answers=total - correct,
        accuracy_percent=round(difficulty_state.get_overall_accuracy() * 100, 1),
        avg_response_time=round(difficulty_state.get_avg_response_time(), 1),
        difficulty_used=state.difficulty,
        mode=state.mode,
        topic_accuracy=topic_acc,
        weak_topics=difficulty_state.get_weak_topics(),
        strong_topics=difficulty_state.get_strong_topics(),
        difficulty_history=difficulty_state.get_difficulty_history(),
        duration_minutes=round(duration, 1),
        question_log=question_log,
        topic_categories=asdict(topic_cats),
        has_study_sheet=False,  # will be set to True after sheet is generated & saved
    )


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def get_historical_weak_topics() -> List[str]:
    """Aggregate persistently weak topics across all past sessions."""
    sessions = list_saved_sessions()
    topic_wrong_count: Dict[str, int] = {}
    topic_total: Dict[str, int] = {}

    for session in sessions:
        for topic, acc in session.get("topic_accuracy", {}).items():
            topic_total[topic] = topic_total.get(topic, 0) + 1
            if acc < 60:
                topic_wrong_count[topic] = topic_wrong_count.get(topic, 0) + 1

    return [
        t for t, total in topic_total.items()
        if total >= 2 and topic_wrong_count.get(t, 0) / total >= 0.5
    ]
