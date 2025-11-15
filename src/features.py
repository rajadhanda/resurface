"""Feature extraction from OCR results."""

import logging
import re
from dataclasses import dataclass
from typing import List

from .ocr import OcrResult

logger = logging.getLogger(__name__)

# Vocabulary lists
MEASURE_UNITS = ["g", "kg", "ml", "l", "tbsp", "tsp", "cup", "cups", "oz", "°c", "°f"]
COOKING_VERBS = [
    "preheat",
    "mix",
    "stir",
    "bake",
    "boil",
    "simmer",
    "chop",
    "fry",
    "whisk",
    "serve",
]
INGREDIENT_SECTION_TERMS = ["ingredients", "serves", "makes", "yield"]
STEP_TERMS = ["instructions", "method", "directions", "steps"]
WORKOUT_TERMS = ["sets", "reps", "rest", "warm-up", "cooldown", "amrap", "emom", "rounds"]
WORKOUT_BODY_PARTS = ["legs", "chest", "back", "shoulders", "glutes", "core", "abs"]
QUOTE_MARKERS = ["—", "-", """, """, '"']


@dataclass
class LayoutFeatures:
    """Layout-based features extracted from OCR lines."""

    line_count: int
    bullet_lines: int
    numbered_lines: int
    avg_line_length: float


@dataclass
class Features:
    """Complete feature set for classification."""

    ocr: OcrResult
    layout: LayoutFeatures
    num_units: int
    num_cooking_verbs: int
    num_workout_terms: int
    num_body_parts: int
    has_ingredients_section: bool
    has_steps_section: bool
    has_quote_author_pattern: bool
    quote_mark_count: int


def compute_layout_features(ocr: OcrResult) -> LayoutFeatures:
    """
    Compute layout features from OCR lines.

    Args:
        ocr: OcrResult containing lines

    Returns:
        LayoutFeatures with counts and averages
    """
    lines = ocr.lines
    line_count = len(lines)

    # Count bullet points (lines starting with •, -, *, etc.)
    bullet_pattern = re.compile(r"^[\s]*[•\-\*\+]\s+")
    bullet_lines = sum(1 for line in lines if bullet_pattern.match(line))

    # Count numbered lines (lines starting with numbers like "1.", "2)", etc.)
    numbered_pattern = re.compile(r"^[\s]*\d+[\.\)]\s+")
    numbered_lines = sum(1 for line in lines if numbered_pattern.match(line))

    # Average line length
    avg_line_length = sum(len(line) for line in lines) / line_count if line_count > 0 else 0.0

    return LayoutFeatures(
        line_count=line_count,
        bullet_lines=bullet_lines,
        numbered_lines=numbered_lines,
        avg_line_length=avg_line_length,
    )


def compute_features(ocr: OcrResult) -> Features:
    """
    Compute all features from OCR result.

    Args:
        ocr: OcrResult from OCR processing

    Returns:
        Features object with all extracted features
    """
    layout = compute_layout_features(ocr)
    text_lower = ocr.full_text.lower()

    # Count units (check if any unit appears in the text)
    num_units = sum(1 for unit in MEASURE_UNITS if unit.lower() in text_lower)

    # Count cooking verbs (check if verb appears in any line)
    num_cooking_verbs = sum(
        1 for verb in COOKING_VERBS if any(verb.lower() in line.lower() for line in ocr.lines)
    )

    # Count workout terms
    num_workout_terms = sum(
        1 for term in WORKOUT_TERMS if any(term.lower() in line.lower() for line in ocr.lines)
    )

    # Count body parts
    num_body_parts = sum(
        1 for part in WORKOUT_BODY_PARTS if any(part.lower() in line.lower() for line in ocr.lines)
    )

    # Check for ingredients section
    has_ingredients_section = any(
        term.lower() in text_lower for term in INGREDIENT_SECTION_TERMS
    )

    # Check for steps section
    has_steps_section = any(term.lower() in text_lower for term in STEP_TERMS)

    # Check for quote author pattern (line starting with "— " or "- " followed by 2+ words)
    quote_author_pattern = re.compile(r"^[\s]*[—\-]\s+\w+\s+\w+")
    has_quote_author_pattern = any(quote_author_pattern.match(line) for line in ocr.lines)

    # Count quote markers
    quote_mark_count = sum(1 for marker in QUOTE_MARKERS if marker in ocr.full_text)

    return Features(
        ocr=ocr,
        layout=layout,
        num_units=num_units,
        num_cooking_verbs=num_cooking_verbs,
        num_workout_terms=num_workout_terms,
        num_body_parts=num_body_parts,
        has_ingredients_section=has_ingredients_section,
        has_steps_section=has_steps_section,
        has_quote_author_pattern=has_quote_author_pattern,
        quote_mark_count=quote_mark_count,
    )

