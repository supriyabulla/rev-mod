"""
feedback_system.py
-------------------
Generates contextual feedback after each answer.
Uses the LLM for rich explanations, with fast local fallbacks.
"""

from dataclasses import dataclass
from typing import Optional
from mcq_generator import MCQuestion
from llm_client import query_llm


@dataclass
class Feedback:
    """Feedback given after an answer attempt."""
    is_correct: bool
    message: str              # Short congratulation or sympathy
    explanation: str          # Why correct answer is right
    reinforcement: str        # Extra insight or memory tip
    needs_followup: bool = False


def generate_feedback(
    question: MCQuestion,
    user_answer_index: int,
    response_time: float,
    use_llm: bool = True,
) -> Feedback:
    """
    Generate feedback for a given answer.
    Falls back to template-based feedback if LLM is slow.
    """
    is_correct = user_answer_index == question.correct_index
    user_answer = question.options[user_answer_index]

    if use_llm:
        return _llm_feedback(question, user_answer, is_correct, response_time)
    else:
        return _template_feedback(question, user_answer, is_correct, response_time)


def _llm_feedback(
    question: MCQuestion,
    user_answer: str,
    is_correct: bool,
    response_time: float,
) -> Feedback:
    """Generate rich feedback using the LLM."""

    speed_note = ""
    if response_time < 8:
        speed_note = "The user answered very quickly."
    elif response_time > 40:
        speed_note = "The user took a long time to answer."

    if is_correct:
        prompt = f"""A student just answered a quiz question CORRECTLY. {speed_note}

Question: {question.question}
Their answer: {user_answer}
Correct answer: {question.options[question.correct_index]}

Provide:
1. A brief encouraging message (1 sentence)
2. A reinforcing explanation that deepens understanding (2-3 sentences)
3. A memory tip or interesting connection (1 sentence)

Return ONLY JSON:
{{"message": "...", "explanation": "...", "reinforcement": "..."}}"""
    else:
        prompt = f"""A student just answered a quiz question INCORRECTLY.

Question: {question.question}
Their answer (WRONG): {user_answer}  
Correct answer: {question.options[question.correct_index]}
Base explanation: {question.explanation}

Provide:
1. A supportive message (1 sentence, not harsh)
2. A clear explanation of why the correct answer is right (2-3 sentences)
3. Why their chosen answer is wrong and a tip to remember (1-2 sentences)

Return ONLY JSON:
{{"message": "...", "explanation": "...", "reinforcement": "..."}}"""

    response = query_llm(prompt, temperature=0.4, max_tokens=300)

    if response:
        import json, re
        response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > 0:
            try:
                data = json.loads(response[start:end])
                return Feedback(
                    is_correct=is_correct,
                    message=data.get("message", ""),
                    explanation=data.get("explanation", question.explanation),
                    reinforcement=data.get("reinforcement", ""),
                    needs_followup=not is_correct,
                )
            except Exception:
                pass

    # Fall through to template if LLM fails
    return _template_feedback(
        question,
        question.options[question.correct_index] if not is_correct else question.options[question.correct_index],
        is_correct,
        response_time,
    )


def _template_feedback(
    question: MCQuestion,
    user_answer: str,
    is_correct: bool,
    response_time: float,
) -> Feedback:
    """Fast template-based feedback without LLM."""

    import random

    correct_messages = [
        "Excellent! That's correct! 🎯",
        "Well done! You got it! ✅",
        "Perfect answer! Keep it up! 💪",
        "Spot on! Great recall! 🌟",
    ]
    wrong_messages = [
        "Not quite — let's review this one. 📖",
        "That wasn't right, but here's the explanation. 🔍",
        "Good try! The correct answer is different. 💡",
        "Let's clarify this concept. 📚",
    ]

    if is_correct:
        speed_reinf = ""
        if response_time < 8:
            speed_reinf = "Great speed too — you clearly know this material well!"
        elif response_time > 40:
            speed_reinf = "You got there! Try to be a bit more decisive next time."

        return Feedback(
            is_correct=True,
            message=random.choice(correct_messages),
            explanation=question.explanation,
            reinforcement=speed_reinf,
            needs_followup=False,
        )
    else:
        return Feedback(
            is_correct=False,
            message=random.choice(wrong_messages),
            explanation=f"The correct answer is: {question.options[question.correct_index]}\n\n{question.explanation}",
            reinforcement=f"Topic to review: {question.topic}",
            needs_followup=True,
        )
