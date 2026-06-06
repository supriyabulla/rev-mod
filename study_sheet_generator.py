"""
study_sheet_generator.py  (NEW — Feature 3)
--------------------------------------------
Generates the end-of-session study sheet from existing in-memory data.

Design principle: reuse ExtractedKnowledge, concepts, key_facts, and the
session question_log as much as possible.  Only falls back to an LLM call
when no structured data is available or when revision notes need synthesis.

Public API (called from server.py):
    generate_study_sheet(session_id, pdf_title, knowledge, question_log,
                         sections) -> StudySheet
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional

from concept_extractor import ExtractedKnowledge, Concept
from pdf_processor import Section
from llm_client import query_llm
from session_manager import StudySheet


# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS for formula detection (no LLM needed for well-known ones)
# ─────────────────────────────────────────────────────────────────────────────
_FORMULA_KEYWORDS = [
    "formula", "equation", "theorem", "law", "rule",
    "=", "∑", "∫", "√", "μ", "σ", "λ", "α", "β",
    "P(", "E(", "Var(", "Cov(",
]

_KNOWN_FORMULAS = {
    "bayes": {"name": "Bayes' Theorem", "formula": "P(A|B) = P(B|A)·P(A) / P(B)", "notes": ""},
    "regression": {"name": "Linear Regression", "formula": "ŷ = β₀ + β₁x", "notes": "β₁ = Cov(x,y)/Var(x)"},
    "chi-square": {"name": "Chi-Square Statistic", "formula": "χ² = Σ (O−E)² / E", "notes": "O=observed, E=expected"},
    "anova": {"name": "ANOVA F-Statistic", "formula": "F = MS_between / MS_within", "notes": ""},
    "standard deviation": {"name": "Standard Deviation", "formula": "σ = √( Σ(xᵢ−μ)² / N )", "notes": ""},
    "variance": {"name": "Variance", "formula": "σ² = Σ(xᵢ−μ)² / N", "notes": ""},
    "z-score": {"name": "Z-Score", "formula": "z = (x − μ) / σ", "notes": ""},
    "poisson": {"name": "Poisson Distribution", "formula": "P(X=k) = (λᵏ · e⁻λ) / k!", "notes": "λ = mean rate"},
    "binomial": {"name": "Binomial Probability", "formula": "P(X=k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ", "notes": ""},
    "central limit": {"name": "Central Limit Theorem", "formula": "x̄ ~ N(μ, σ²/n) as n→∞", "notes": ""},
    "correlation": {"name": "Pearson Correlation", "formula": "r = Cov(X,Y) / (σₓ · σᵧ)", "notes": ""},
    "confidence interval": {"name": "Confidence Interval", "formula": "CI = x̄ ± z*(σ/√n)", "notes": ""},
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_study_sheet(
    session_id: str,
    pdf_title: str,
    knowledge: ExtractedKnowledge,
    question_log: List[dict],
    sections: List[Section],
) -> StudySheet:
    """
    Build a complete StudySheet from session data.
    Steps:
      1. definitions — from ExtractedKnowledge.concepts (NO LLM call)
      2. keywords    — from concepts + question topics (NO LLM call)
      3. formulas    — keyword-matched from content (NO LLM call in most cases)
      4. revision_notes — LLM synthesis from section_summaries + weak topics
    """
    sheet = StudySheet(
        session_id=session_id,
        pdf_title=pdf_title,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ── 1. Key Definitions (reuse concepts directly) ──────────────────────────
    sheet.definitions = _extract_definitions(knowledge.concepts)

    # ── 2. Important Keywords ─────────────────────────────────────────────────
    sheet.keywords = _extract_keywords(knowledge, question_log)

    # ── 3. Formulas ───────────────────────────────────────────────────────────
    sheet.formulas = _detect_formulas(knowledge, sections)

    # ── 4. Revision Notes (LLM call) ─────────────────────────────────────────
    sheet.revision_notes = _generate_revision_notes(knowledge, question_log, sections)

    return sheet


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Definitions from existing Concept objects
# ─────────────────────────────────────────────────────────────────────────────
def _extract_definitions(concepts: List[Concept]) -> List[Dict[str, str]]:
    """
    Convert Concept objects into definition entries.
    Deduplicate by name (case-insensitive). Cap at 25.
    """
    seen = set()
    defs = []
    for c in concepts:
        key = c.name.lower().strip()
        if key and key not in seen and c.definition:
            seen.add(key)
            defs.append({
                "term": c.name,
                "definition": c.definition,
            })
    return defs[:25]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Keywords from concepts + question topics
# ─────────────────────────────────────────────────────────────────────────────
def _extract_keywords(
    knowledge: ExtractedKnowledge,
    question_log: List[dict],
) -> List[str]:
    """
    Build keyword list from:
      • concept names
      • topic names seen in question_log
      • related_topics from concepts
    Deduplicate, sort, cap at 40.
    """
    keywords = set()

    # From concept names
    for c in knowledge.concepts:
        if c.name and len(c.name) > 2:
            keywords.add(c.name.strip())
        for rt in c.related_topics:
            if rt and len(rt) > 2:
                keywords.add(rt.strip())

    # From question topics
    for entry in question_log:
        topic = entry.get("topic", "")
        if topic and len(topic) > 2 and topic != "Key Facts":
            keywords.add(topic.strip())

    return sorted(keywords)[:40]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Formula detection from content (no LLM)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_formulas(
    knowledge: ExtractedKnowledge,
    sections: List[Section],
) -> List[Dict[str, str]]:
    """
    Check all content for known formula keywords.
    Returns matched entries from _KNOWN_FORMULAS.
    Falls back to LLM extraction if content looks math-heavy.
    """
    found_formulas = []
    seen_names = set()

    # Collect all text to search
    all_text = " ".join(knowledge.key_facts).lower()
    for s in sections:
        all_text += " " + s.content.lower()
    for c in knowledge.concepts:
        all_text += " " + c.name.lower() + " " + c.definition.lower()

    # Match against known formula keywords
    for keyword, formula_data in _KNOWN_FORMULAS.items():
        if keyword in all_text and formula_data["name"] not in seen_names:
            seen_names.add(formula_data["name"])
            found_formulas.append(dict(formula_data))  # copy to avoid mutation

    # If content looks math-heavy but no matches found, try LLM extraction
    formula_signal = sum(1 for kw in _FORMULA_KEYWORDS if kw in all_text)
    if formula_signal >= 4 and not found_formulas:
        llm_formulas = _llm_extract_formulas(knowledge, sections)
        found_formulas.extend(llm_formulas)

    return found_formulas[:10]


def _llm_extract_formulas(
    knowledge: ExtractedKnowledge,
    sections: List[Section],
) -> List[Dict[str, str]]:
    """
    Fallback: ask LLM to extract formulas when keyword matching finds nothing
    but content looks mathematical.
    """
    # Use section summaries as condensed input (cheaper than full content)
    summaries_text = "\n".join(
        f"- {title}: {summary}"
        for title, summary in list(knowledge.section_summaries.items())[:8]
    )

    prompt = f"""From these study notes, extract any mathematical formulas, equations or statistical rules.

Notes:
{summaries_text}

Return ONLY valid JSON array (no markdown):
[
  {{"name": "Formula Name", "formula": "symbolic formula", "notes": "brief context"}}
]

If no formulas exist, return: []
Maximum 8 formulas."""

    response = query_llm(prompt, temperature=0.1, max_tokens=500)
    if not response:
        return []

    # Parse JSON array
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
    start = response.find("[")
    end = response.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        items = json.loads(response[start:end])
        return [i for i in items if isinstance(i, dict) and i.get("name") and i.get("formula")][:8]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Revision notes via LLM (one call, synthesises session material)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_revision_notes(
    knowledge: ExtractedKnowledge,
    question_log: List[dict],
    sections: List[Section],
) -> List[str]:
    """
    Generate concise revision bullet points.
    Input: section summaries + topics that appeared in the quiz.
    Uses a single LLM call. Falls back to key_facts if LLM fails.
    """
    # Determine which topics were actually tested in this session
    tested_topics = list({
        entry.get("topic", "") for entry in question_log
        if entry.get("topic") and entry["topic"] != "Key Facts"
    })

    # Identify wrong-answer topics for extra emphasis
    wrong_topics = list({
        entry.get("topic", "") for entry in question_log
        if not entry.get("is_correct") and entry.get("topic")
    })

    # Build a compact input for the LLM from section summaries
    summaries_text = "\n".join(
        f"• {title}: {summary}"
        for title, summary in list(knowledge.section_summaries.items())[:10]
    )

    # Add key facts as extra context (no extra LLM call — already extracted)
    facts_text = "\n".join(f"• {f}" for f in knowledge.key_facts[:15])

    # Flag weak topics so LLM emphasises them
    weak_note = ""
    if wrong_topics:
        weak_note = f"\nPay extra attention to these topics (answered incorrectly): {', '.join(wrong_topics[:6])}"

    prompt = f"""You are a study notes generator. Based on these study summaries and key facts,
generate concise revision bullet points a student can use to quickly review before an exam.

Section summaries:
{summaries_text}

Key facts:
{facts_text}
{weak_note}

Generate 8-15 bullet points. Each bullet should:
- Be one clear, memorable sentence
- Capture an important concept or distinction
- Be exam-focused

Return ONLY a JSON array of strings (no markdown, no numbering):
["bullet point 1", "bullet point 2", ...]"""

    response = query_llm(prompt, temperature=0.3, max_tokens=700)

    if response:
        response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > 0:
            try:
                notes = json.loads(response[start:end])
                if isinstance(notes, list) and notes:
                    return [str(n) for n in notes if n][:15]
            except Exception:
                pass

    # Fallback: return key_facts as revision notes
    return knowledge.key_facts[:12]
