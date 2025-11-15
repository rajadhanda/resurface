"""Phase 0 heuristic classifier for screenshots."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Literal

from .features import Features, QUOTE_MARKERS

logger = logging.getLogger(__name__)

ItemType = Literal["recipe", "workout", "quote", "none"]


@dataclass
class ClassificationResult:
    """Result of classification with scores and chosen type."""

    item_type: ItemType
    scores: Dict[str, float]
    threshold: float


def score_recipe(feats: Features) -> float:
    """
    Compute recipe score based on features.

    Args:
        feats: Features object

    Returns:
        Recipe score (higher = more likely to be a recipe)
    """
    score = 0.0

    # +3 if any INGREDIENT_SECTION_TERMS appears
    if feats.has_ingredients_section:
        score += 3.0

    # +2 if num_units >= 3
    if feats.num_units >= 3:
        score += 2.0

    # +2 if num_cooking_verbs >= 2
    if feats.num_cooking_verbs >= 2:
        score += 2.0

    # +1 if bullet_lines >= 3
    if feats.layout.bullet_lines >= 3:
        score += 1.0

    # +1 if numbered_lines >= 2
    if feats.layout.numbered_lines >= 2:
        score += 1.0

    # +1 if any line matches "serves X" or "makes X"
    serves_pattern = re.compile(r"(serves|makes)\s+\d+", re.IGNORECASE)
    if any(serves_pattern.search(line) for line in feats.ocr.lines):
        score += 1.0

    return score


def score_workout(feats: Features) -> float:
    """
    Compute workout score based on features.

    Args:
        feats: Features object

    Returns:
        Workout score (higher = more likely to be a workout)
    """
    score = 0.0

    # +3 if "sets" or "reps" appears in multiple lines
    sets_reps_count = sum(
        1
        for line in feats.ocr.lines
        if "sets" in line.lower() or "reps" in line.lower()
    )
    if sets_reps_count >= 2:
        score += 3.0

    # +2 if a pattern like 3x10 / 3 x 10 occurs
    workout_pattern = re.compile(r"\d+\s*[x×]\s*\d+", re.IGNORECASE)
    if any(workout_pattern.search(line) for line in feats.ocr.lines):
        score += 2.0

    # +2 if num_workout_terms >= 2
    if feats.num_workout_terms >= 2:
        score += 2.0

    # +1 if any WORKOUT_BODY_PARTS appears
    if feats.num_body_parts > 0:
        score += 1.0

    # +1 if there are bullet/numbered lines suggesting a list
    if feats.layout.bullet_lines >= 2 or feats.layout.numbered_lines >= 2:
        score += 1.0

    # Subtract 2 if has_ingredients_section is True (to avoid misclassifying recipes as workouts)
    if feats.has_ingredients_section:
        score -= 2.0

    return score


def score_quote(feats: Features) -> float:
    """
    Compute quote score based on features.

    Args:
        feats: Features object

    Returns:
        Quote score (higher = more likely to be a quote)
    """
    score = 0.0

    # +2 if any line contains quote marks and 4+ words
    for line in feats.ocr.lines:
        word_count = len(line.split())
        has_quote_mark = any(marker in line for marker in QUOTE_MARKERS)
        if has_quote_mark and word_count >= 4:
            score += 2.0
            break  # Only count once

    # +3 if has_quote_author_pattern
    if feats.has_quote_author_pattern:
        score += 3.0

    # +1 if 1 <= line_count <= 6
    if 1 <= feats.layout.line_count <= 6:
        score += 1.0

    # +1 if avg_line_length is relatively high (prose) - threshold at 40 chars
    if feats.layout.avg_line_length >= 40:
        score += 1.0

    # +1 if num_units == 0 and num_workout_terms == 0
    if feats.num_units == 0 and feats.num_workout_terms == 0:
        score += 1.0

    return score


def classify(feats: Features, threshold: float = 5.0) -> ClassificationResult:
    """
    Compute recipe/workout/quote scores, pick the best type if above threshold.

    Otherwise return item_type="none".

    Args:
        feats: Features object
        threshold: Minimum score to classify as a specific type (default: 5.0)

    Returns:
        ClassificationResult with item_type, scores, and threshold
    """
    scores = {
        "recipe": score_recipe(feats),
        "workout": score_workout(feats),
        "quote": score_quote(feats),
    }

    # Find the best type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # If best score < threshold, return "none"
    if best_score < threshold:
        item_type: ItemType = "none"
    else:
        item_type = best_type  # type: ignore

    logger.debug(
        f"Classification: {item_type} (scores: {scores}, threshold: {threshold})"
    )

    return ClassificationResult(
        item_type=item_type,
        scores=scores,
        threshold=threshold,
    )

