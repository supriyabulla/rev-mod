"""
mcq_generator.py  (v3)
-----------------------
Changes from v2:
  • _build_source_pool now:
      - skips example sections entirely (_is_example_section check)
      - skips questions whose hash is in the flagged set
      - deprioritises topics/sections with multiple flags (moved to end of pool)
  • LLM generation prompt has explicit "DO NOT use examples" instruction
  • generate_questions_for_session accepts optional flagged_hashes + flagged_topics
  • MCQuestion gets a `question_hash` field for flag deduplication
"""

import json
import re
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
from concept_extractor import ExtractedKnowledge, Concept, _is_example_section
from pdf_processor import Section
from llm_client import query_llm


@dataclass
class MCQuestion:
    """A fully formed MCQ."""
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    difficulty: str
    question_type: str
    topic: str
    section_title: str = ""
    question_hash: str = ""   # v3: for flag deduplication

    def __post_init__(self):
        if not self.question_hash and self.question:
            self.question_hash = hashlib.md5(
                self.question.strip().lower().encode()
            ).hexdigest()[:12]

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]

    @property
    def correct_letter(self) -> str:
        return "ABCD"[self.correct_index]


DIFFICULTY_PROMPTS = {
    "easy": """Generate an EASY multiple choice question:
- Tests recall of a basic definition or principle
- Clear, direct question with unambiguous correct answer
- Distractors are plausible but clearly wrong to anyone who studied
- No tricks, no negation""",

    "medium": """Generate a MEDIUM difficulty multiple choice question:
- Tests understanding of a concept, not just recall
- Requires comparing ideas or applying a principle
- Distractors test common misconceptions
- May use "which is TRUE / FALSE" style""",

    "hard": """Generate a HARD multiple choice question:
- Tests deep analysis, synthesis, or application of a concept
- Distractors are very close to the correct answer
- May use negation (which is NOT), edge cases, or subtle distinctions
- Requires genuine mastery""",
}

QUESTION_TYPE_PROMPTS = {
    "definition":  "The question MUST test the definition or precise meaning of a key term.",
    "conceptual":  "The question MUST test understanding of a concept or principle.",
    "application": "The question MUST ask how a concept applies to a general scenario.",
    "trick":       "The question MUST use careful wording (e.g. 'which is NOT', 'EXCEPT').",
}

# Instruction appended to every generation prompt to block example-based questions
_NO_EXAMPLES_INSTRUCTION = """
CRITICAL: Do NOT base the question on a specific numerical example, worked example,
case study, or illustration from the text. The question must test the GENERAL CONCEPT
or PRINCIPLE, not a detail from an example. If the source material only contains
examples, generate a question about the underlying concept those examples demonstrate."""


def generate_questions_for_session(
    knowledge: ExtractedKnowledge,
    sections: List[Section],
    difficulty: str = "medium",
    count: int = 10,
    flagged_hashes: Optional[Set[str]] = None,
    flagged_topics: Optional[Dict[str, int]] = None,
) -> List[MCQuestion]:
    """
    Generate a batch of MCQs for a study session.
    v3: skips flagged questions and example-derived content.
    """
    flagged_hashes = flagged_hashes or set()
    flagged_topics = flagged_topics or {}

    questions: List[MCQuestion] = []
    qtypes = ["definition", "conceptual", "application", "conceptual", "trick"]

    sources = _build_source_pool(
        knowledge, sections, flagged_topics=flagged_topics
    )
    if not sources:
        return []

    random.shuffle(sources)
    print(f"\n⚙️  Generating {count} questions at {difficulty.upper()} difficulty...")

    generated = 0
    attempts = 0
    max_attempts = count * 4  # more attempts to account for flagged skips

    while generated < count and attempts < max_attempts:
        attempts += 1
        source = sources[attempts % len(sources)]
        qtype = qtypes[generated % len(qtypes)]

        q = _generate_single_question(
            content=source["content"],
            topic=source["topic"],
            section_title=source["section"],
            difficulty=difficulty,
            qtype=qtype,
        )
        if not q:
            continue

        # Skip if this exact question was previously flagged
        if q.question_hash in flagged_hashes:
            continue

        questions.append(q)
        generated += 1
        print(f"   ✓ Question {generated}/{count}".ljust(40), end="\r")

    print(f"\n✅ Generated {len(questions)} questions.")
    return questions


def generate_followup_question(
    original_question: MCQuestion,
    knowledge: ExtractedKnowledge,
    sections: List[Section],
) -> Optional[MCQuestion]:
    section_content = ""
    for s in sections:
        if (s.title == original_question.section_title or
                original_question.topic.lower() in s.content.lower()):
            if not _is_example_section(s):
                section_content = s.content[:600]
                break

    if not section_content:
        section_content = f"Topic: {original_question.topic}"

    return _generate_single_question(
        content=section_content,
        topic=original_question.topic,
        section_title=original_question.section_title,
        difficulty="easy",
        qtype="definition",
    )


def _build_source_pool(
    knowledge: ExtractedKnowledge,
    sections: List[Section],
    flagged_topics: Dict[str, int] = None,
) -> list:
    """
    Build the question source pool.
    v3 changes:
      - example sections excluded entirely
      - heavily flagged topics moved to end of pool (deprioritised, not removed)
      - key_facts excluded from pool (they're too loose; concepts are used instead)
    """
    flagged_topics = flagged_topics or {}
    normal_pool = []
    deprioritised_pool = []

    # From extracted concepts (primary source — most precise)
    for concept in knowledge.concepts:
        if concept.definition:
            entry = {
                "content": f"{concept.name}: {concept.definition}",
                "topic": concept.name,
                "section": concept.section_title or "General",
            }
            if flagged_topics.get(concept.name, 0) >= 2:
                deprioritised_pool.append(entry)
            else:
                normal_pool.append(entry)

    # From section content — skip example sections
    for section in sections:
        if section.word_count > 80 and not _is_example_section(section):
            entry = {
                "content": section.content[:500],
                "topic": section.title,
                "section": section.title,
            }
            if flagged_topics.get(section.title, 0) >= 2:
                deprioritised_pool.append(entry)
            else:
                normal_pool.append(entry)

    # NOTE: key_facts deliberately excluded — too imprecise for question generation.
    # Concepts and section content provide better grounding.

    return normal_pool + deprioritised_pool


def _generate_single_question(
    content: str,
    topic: str,
    section_title: str,
    difficulty: str,
    qtype: str,
) -> Optional[MCQuestion]:
    diff_prompt = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["medium"])
    type_prompt = QUESTION_TYPE_PROMPTS.get(qtype, QUESTION_TYPE_PROMPTS["conceptual"])

    prompt = f"""{diff_prompt}

{type_prompt}
{_NO_EXAMPLES_INSTRUCTION}

Source material:
\"\"\"{content}\"\"\"

Return ONLY valid JSON (no markdown):
{{
  "question": "The question text?",
  "options": ["A) option", "B) option", "C) option", "D) option"],
  "correct_index": 0,
  "explanation": "Why the correct answer is right.",
  "topic": "{topic}"
}}

Rules:
- correct_index is 0-based (0=A, 1=B, 2=C, 3=D)
- All 4 options required, starting with "A) " "B) " "C) " "D) "
- Question must end with ?
- No specific numbers or names from examples"""

    response = query_llm(prompt, temperature=0.75, max_tokens=600)
    if not response:
        return None

    parsed = _parse_mcq_json(response)
    if not parsed:
        return None

    return MCQuestion(
        question=parsed.get("question", ""),
        options=parsed.get("options", []),
        correct_index=int(parsed.get("correct_index", 0)),
        explanation=parsed.get("explanation", ""),
        difficulty=difficulty,
        question_type=qtype,
        topic=parsed.get("topic", topic),
        section_title=section_title,
    )


def _parse_mcq_json(response: str) -> Optional[dict]:
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    json_str = re.sub(r",\s*([}\]])", r"\1", response[start:end])
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not data.get("question"):
        return None
    options = data.get("options", [])
    if len(options) != 4:
        return None
    ci = data.get("correct_index", 0)
    try:
        ci = int(ci)
        if not (0 <= ci <= 3):
            ci = 0
    except (ValueError, TypeError):
        ci = 0
    data["correct_index"] = ci
    return data


def shuffle_options(question: MCQuestion) -> MCQuestion:
    correct_text = question.options[question.correct_index]
    shuffled = question.options.copy()
    random.shuffle(shuffled)
    new_correct_index = shuffled.index(correct_text)
    return MCQuestion(
        question=question.question,
        options=shuffled,
        correct_index=new_correct_index,
        explanation=question.explanation,
        difficulty=question.difficulty,
        question_type=question.question_type,
        topic=question.topic,
        section_title=question.section_title,
        question_hash=question.question_hash,
    )
