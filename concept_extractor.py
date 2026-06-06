"""
concept_extractor.py  (v3)
---------------------------
Changes from v2:
  • LLM prompt now explicitly instructs to SKIP examples, worked examples,
    case studies and numerical illustrations when extracting facts.
  • Facts are now extracted as precise exam-worthy statements (not loose notes).
  • _is_example_section() heuristic pre-filters obvious example sections before
    sending to LLM, saving tokens.
  • key_facts capped at 40 (was 80) — precision over volume.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pdf_processor import Section
from llm_client import query_llm


@dataclass
class Concept:
    """A key concept extracted from the document."""
    name: str
    definition: str
    related_topics: List[str] = field(default_factory=list)
    section_title: str = ""
    importance: str = "medium"


@dataclass
class ExtractedKnowledge:
    """All structured knowledge from the document."""
    concepts: List[Concept]
    key_facts: List[str]
    section_summaries: Dict[str, str]
    topic_map: Dict[str, List[str]]


# ── Example-section detection ─────────────────────────────────────────────────
# These patterns in section titles or content signal example material that
# should NOT be used as question sources.

_EXAMPLE_TITLE_PATTERNS = re.compile(
    r"\b(example|worked example|illustration|case study|eg\.|e\.g\.|"
    r"sample problem|practice problem|solution|exhibit \d|figure \d|"
    r"table \d|appendix)\b",
    re.IGNORECASE,
)

_EXAMPLE_CONTENT_SIGNALS = [
    "for example,", "for instance,", "as an example",
    "consider the following example", "let's say", "let us say",
    "suppose that", "imagine that", "in this example",
    "worked example", "sample calculation",
]


def _is_example_section(section: Section) -> bool:
    """
    Returns True if a section is primarily example/illustration material.
    Used to skip it during question generation source pool building.
    """
    title_lower = section.title.lower()
    if _EXAMPLE_TITLE_PATTERNS.search(title_lower):
        return True

    # Check first 200 words of content for example signals
    snippet = " ".join(section.content.split()[:200]).lower()
    signal_count = sum(1 for sig in _EXAMPLE_CONTENT_SIGNALS if sig in snippet)

    # If 2+ example signals in the opening, treat as example section
    return signal_count >= 2


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_concepts_from_sections(sections: List[Section]) -> ExtractedKnowledge:
    """
    Process each section through the LLM to extract structured knowledge.
    Example sections are identified and skipped for fact/concept extraction.
    """
    all_concepts: List[Concept] = []
    all_facts: List[str] = []
    section_summaries: Dict[str, str] = {}
    topic_map: Dict[str, List[str]] = {}

    print(f"\n🔍 Analyzing {len(sections)} sections...")

    for i, section in enumerate(sections):
        print(f"   Processing: {section.title[:50]}... ({i+1}/{len(sections)})".ljust(72), end="\r")

        if section.word_count < 30:
            continue

        # Skip example sections — don't extract facts/concepts from them
        if _is_example_section(section):
            print(f"   Skipping example section: {section.title[:40]}".ljust(72), end="\r")
            continue

        content_snippet = _truncate(section.content, max_words=600)
        extraction = _extract_from_section(section.title, content_snippet)

        if extraction:
            all_concepts.extend(extraction.get("concepts", []))
            all_facts.extend(extraction.get("facts", []))
            summary = extraction.get("summary", "")
            if summary:
                section_summaries[section.title] = summary

    print("\n✅ Knowledge extraction complete.".ljust(72))

    topic_map = _build_topic_map(all_concepts)
    all_facts = list(dict.fromkeys(all_facts))  # deduplicate preserving order
    concept_objects = _dicts_to_concepts(all_concepts, sections)

    return ExtractedKnowledge(
        concepts=concept_objects,
        key_facts=all_facts[:40],   # v3: reduced to 40, higher precision
        section_summaries=section_summaries,
        topic_map=topic_map,
    )


def _extract_from_section(title: str, content: str) -> Optional[dict]:
    """
    Ask the LLM to extract structured knowledge.
    v3: explicit instruction to ignore examples, numbers, and illustrations.
    """
    prompt = f"""You are an expert educator extracting exam-ready knowledge from a textbook section.

Section Title: {title}

Content:
{content}

IMPORTANT RULES:
- IGNORE any worked examples, numerical illustrations, case studies, or "for example" passages.
- Extract only general CONCEPTS and PRINCIPLES, not specific numbers from examples.
- Facts must be precise, exam-worthy statements of principle — not observations from examples.
- A good fact states a rule, definition, relationship, or property that holds universally.
- A bad fact describes what happened in a specific example ("In Example 3, X was 42").

Return ONLY a valid JSON object (no markdown):
{{
  "summary": "2-3 sentence conceptual summary (no example details)",
  "concepts": [
    {{"name": "concept name", "definition": "precise definition", "related": ["topic1", "topic2"]}}
  ],
  "facts": [
    "A precise, universally true statement about the concept",
    "Another exam-worthy principle or rule"
  ]
}}

Extract 2-5 concepts and 3-6 precise facts. No example-specific content."""

    response = query_llm(prompt, temperature=0.2, max_tokens=800)
    if not response:
        return None
    return _parse_json_response(response)


def _parse_json_response(response: str) -> Optional[dict]:
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", response[start:end])
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _dicts_to_concepts(raw: list, sections: List[Section]) -> List[Concept]:
    concepts = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            concepts.append(Concept(
                name=str(item.get("name", ""))[:100],
                definition=str(item.get("definition", ""))[:500],
                related_topics=item.get("related", [])[:5],
                importance="medium",
            ))
    return concepts


def _build_topic_map(concepts: list) -> Dict[str, List[str]]:
    topic_map: Dict[str, List[str]] = {}
    for c in concepts:
        if isinstance(c, dict):
            name = c.get("name", "")
            related = c.get("related", [])
        else:
            name = c.name
            related = c.related_topics
        if name:
            topic_map[name] = related
    return topic_map


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."
