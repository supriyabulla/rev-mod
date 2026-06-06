"""
difficulty_engine.py
---------------------
Manages adaptive difficulty adjustments based on user performance.
Tracks accuracy, response time, and topic-level performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AnswerRecord:
    """Records a single answer attempt."""
    question_num: int
    topic: str
    difficulty: str
    was_correct: bool
    response_time_seconds: float
    section_title: str = ""


@dataclass
class DifficultyState:
    """Tracks the current difficulty state and performance metrics."""
    current_difficulty: str = Difficulty.MEDIUM
    mode: str = "manual"            # "manual" or "adaptive"
    history: List[AnswerRecord] = field(default_factory=list)

    # Window for adaptive decisions (last N questions)
    window_size: int = 5

    # Per-topic tracking
    topic_correct: Dict[str, int] = field(default_factory=dict)
    topic_total: Dict[str, int] = field(default_factory=dict)

    # Scoring thresholds (seconds)
    fast_threshold: float = 15.0   # <= this = fast response
    slow_threshold: float = 45.0   # >= this = slow response

    # Difficulty promotion thresholds
    promote_accuracy: float = 0.80  # 80%+ correct in window → go harder
    demote_accuracy: float = 0.40   # <40% correct in window → go easier

    def record_answer(
        self,
        question_num: int,
        topic: str,
        was_correct: bool,
        response_time: float,
        section_title: str = "",
    ) -> Optional[str]:
        """
        Record an answer and return a difficulty change message if applicable.
        """
        record = AnswerRecord(
            question_num=question_num,
            topic=topic,
            difficulty=self.current_difficulty,
            was_correct=was_correct,
            response_time_seconds=response_time,
            section_title=section_title,
        )
        self.history.append(record)

        # Update topic stats
        self.topic_total[topic] = self.topic_total.get(topic, 0) + 1
        if was_correct:
            self.topic_correct[topic] = self.topic_correct.get(topic, 0) + 1

        # Only adjust if in adaptive mode
        if self.mode == "adaptive":
            return self._maybe_adjust_difficulty(was_correct, response_time)
        return None

    def _maybe_adjust_difficulty(
        self, was_correct: bool, response_time: float
    ) -> Optional[str]:
        """
        Adaptive logic:
        - Wrong → demote (or keep easy)
        - Fast + correct for window → promote
        - Slow + correct → keep same
        - Low window accuracy → demote
        """
        # Immediate demotion on wrong answer
        if not was_correct:
            new_diff = self._demote()
            if new_diff:
                return f"📉 Difficulty adjusted to {new_diff.upper()} — let's reinforce the basics."
            return None

        # Evaluate rolling window
        window = self.history[-self.window_size:]
        if len(window) < 3:
            return None  # Not enough data yet

        window_accuracy = sum(1 for r in window if r.was_correct) / len(window)
        avg_time = sum(r.response_time_seconds for r in window) / len(window)

        if window_accuracy >= self.promote_accuracy and avg_time <= self.fast_threshold:
            new_diff = self._promote()
            if new_diff:
                return f"📈 Great pace! Difficulty raised to {new_diff.upper()}."
        elif window_accuracy < self.demote_accuracy:
            new_diff = self._demote()
            if new_diff:
                return f"📉 Difficulty reduced to {new_diff.upper()} — accuracy is low."

        return None

    def _promote(self) -> Optional[str]:
        """Move up one difficulty level. Returns new level or None if already at max."""
        order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
        current_idx = order.index(Difficulty(self.current_difficulty))
        if current_idx < len(order) - 1:
            self.current_difficulty = order[current_idx + 1]
            return self.current_difficulty
        return None

    def _demote(self) -> Optional[str]:
        """Move down one difficulty level. Returns new level or None if already at min."""
        order = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
        current_idx = order.index(Difficulty(self.current_difficulty))
        if current_idx > 0:
            self.current_difficulty = order[current_idx - 1]
            return self.current_difficulty
        return None

    def get_weak_topics(self, min_attempts: int = 2) -> List[str]:
        """Return topics where accuracy is below 60%."""
        weak = []
        for topic, total in self.topic_total.items():
            if total >= min_attempts:
                correct = self.topic_correct.get(topic, 0)
                accuracy = correct / total
                if accuracy < 0.6:
                    weak.append(topic)
        return weak

    def get_strong_topics(self, min_attempts: int = 2) -> List[str]:
        """Return topics where accuracy is 80%+."""
        strong = []
        for topic, total in self.topic_total.items():
            if total >= min_attempts:
                correct = self.topic_correct.get(topic, 0)
                accuracy = correct / total
                if accuracy >= 0.8:
                    strong.append(topic)
        return strong

    def get_window_accuracy(self) -> float:
        """Accuracy over the last N questions."""
        window = self.history[-self.window_size:]
        if not window:
            return 0.0
        return sum(1 for r in window if r.was_correct) / len(window)

    def get_overall_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.was_correct) / len(self.history)

    def get_avg_response_time(self) -> float:
        if not self.history:
            return 0.0
        return sum(r.response_time_seconds for r in self.history) / len(self.history)

    def get_topic_accuracy_map(self) -> Dict[str, float]:
        """Return accuracy per topic."""
        result = {}
        for topic, total in self.topic_total.items():
            correct = self.topic_correct.get(topic, 0)
            result[topic] = round(correct / total * 100, 1) if total > 0 else 0.0
        return result

    def get_difficulty_history(self) -> List[str]:
        """Return list of difficulty levels for each question."""
        return [r.difficulty for r in self.history]
