"""
question_flags.py  (NEW — v3)
------------------------------
Persistent store for flagged questions.

When a student flags a question as "wrong/bad", we:
  1. Save it to ~/.study_assistant/flags.json
  2. Record: question text hash, question text, topic, section, reason, timestamp
  3. Expose load_flags() so mcq_generator can exclude flagged content
  4. Expose get_flag_stats() so future sessions can avoid similar source material

File: ~/.study_assistant/flags.json
Format: { "flagged": [ {FlagRecord}, ... ] }
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

FLAGS_FILE = Path.home() / ".study_assistant" / "flags.json"


@dataclass
class FlagRecord:
    """A single flagged question record."""
    question_hash: str       # md5 of question text — dedup key
    question_text: str
    options: List[str]
    topic: str
    section_title: str
    difficulty: str
    reason: str              # "wrong_question" | "unclear" | "example_based" | "other"
    flagged_at: str          # ISO datetime
    session_id: str = ""


def _hash_question(question_text: str) -> str:
    return hashlib.md5(question_text.strip().lower().encode()).hexdigest()[:12]


def load_flags() -> List[FlagRecord]:
    """Load all flagged questions from disk."""
    if not FLAGS_FILE.exists():
        return []
    try:
        with open(FLAGS_FILE) as f:
            data = json.load(f)
        return [FlagRecord(**r) for r in data.get("flagged", [])]
    except Exception:
        return []


def save_flag(
    question_text: str,
    options: List[str],
    topic: str,
    section_title: str,
    difficulty: str,
    reason: str,
    session_id: str = "",
) -> FlagRecord:
    """
    Add a flagged question to the persistent store.
    Deduplicates by question hash — same question flagged twice = one record.
    Returns the saved FlagRecord.
    """
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = load_flags()
    qhash = _hash_question(question_text)

    # Dedup — if already flagged, just update the reason and timestamp
    for rec in existing:
        if rec.question_hash == qhash:
            rec.reason = reason
            rec.flagged_at = datetime.now().isoformat()
            _write_flags(existing)
            return rec

    new_flag = FlagRecord(
        question_hash=qhash,
        question_text=question_text,
        options=options,
        topic=topic,
        section_title=section_title,
        difficulty=difficulty,
        reason=reason,
        flagged_at=datetime.now().isoformat(),
        session_id=session_id,
    )
    existing.append(new_flag)
    _write_flags(existing)
    return new_flag


def _write_flags(flags: List[FlagRecord]):
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_FILE, "w") as f:
        json.dump({"flagged": [asdict(r) for r in flags]}, f, indent=2)


def get_flagged_hashes() -> set:
    """Return set of question_hash values — used to skip flagged questions."""
    return {r.question_hash for r in load_flags()}


def get_flagged_topics() -> Dict[str, int]:
    """
    Return dict of topic → flag count.
    Topics with multiple flags get deprioritised in future question generation.
    """
    counts: Dict[str, int] = {}
    for rec in load_flags():
        counts[rec.topic] = counts.get(rec.topic, 0) + 1
    return counts


def get_flagged_sections() -> Dict[str, int]:
    """Return dict of section_title → flag count."""
    counts: Dict[str, int] = {}
    for rec in load_flags():
        if rec.section_title:
            counts[rec.section_title] = counts.get(rec.section_title, 0) + 1
    return counts


def is_question_flagged(question_text: str) -> bool:
    """Quick check — is this specific question already flagged?"""
    qhash = _hash_question(question_text)
    return qhash in get_flagged_hashes()


def get_flag_summary() -> dict:
    """Summary stats for the UI."""
    flags = load_flags()
    return {
        "total_flagged": len(flags),
        "by_reason": _count_by(flags, "reason"),
        "by_topic": _count_by(flags, "topic"),
        "recent": [asdict(f) for f in flags[-5:]],
    }


def _count_by(flags: List[FlagRecord], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in flags:
        val = getattr(rec, field_name, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts
