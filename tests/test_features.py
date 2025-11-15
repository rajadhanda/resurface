"""Tests for feature extraction."""

import pytest
from src.features import (
    LayoutFeatures,
    Features,
    compute_layout_features,
    compute_features,
)
from src.ocr import OcrResult


def test_compute_layout_features():
    """Test layout feature computation."""
    # Test with empty OCR
    ocr_empty = OcrResult(full_text="", lines=[])
    layout_empty = compute_layout_features(ocr_empty)
    assert layout_empty.line_count == 0
    assert layout_empty.bullet_lines == 0
    assert layout_empty.numbered_lines == 0
    assert layout_empty.avg_line_length == 0.0

    # Test with sample lines
    ocr_sample = OcrResult(
        full_text="• Item 1\n• Item 2\n1. First step\n2. Second step\nRegular text line",
        lines=[
            "• Item 1",
            "• Item 2",
            "1. First step",
            "2. Second step",
            "Regular text line",
        ],
    )
    layout = compute_layout_features(ocr_sample)
    assert layout.line_count == 5
    assert layout.bullet_lines == 2
    assert layout.numbered_lines == 2
    assert layout.avg_line_length > 0


def test_compute_features():
    """Test full feature computation."""
    # Test with recipe-like text
    ocr_recipe = OcrResult(
        full_text="Ingredients\n2 cups flour\n1 tbsp sugar\nMix and bake at 350°F",
        lines=[
            "Ingredients",
            "2 cups flour",
            "1 tbsp sugar",
            "Mix and bake at 350°F",
        ],
    )
    features = compute_features(ocr_recipe)

    assert features.layout.line_count == 4
    assert features.num_units >= 2  # "cups", "tbsp", "°F"
    assert features.num_cooking_verbs >= 1  # "Mix", "bake"
    assert features.has_ingredients_section is True
    assert features.num_workout_terms == 0
    assert features.num_body_parts == 0

    # Test with workout-like text
    ocr_workout = OcrResult(
        full_text="3 sets of 10 reps\nRest 60 seconds\nLegs workout",
        lines=[
            "3 sets of 10 reps",
            "Rest 60 seconds",
            "Legs workout",
        ],
    )
    features_workout = compute_features(ocr_workout)

    assert features_workout.num_workout_terms >= 2  # "sets", "reps", "rest"
    assert features_workout.num_body_parts >= 1  # "Legs"
    assert features_workout.has_ingredients_section is False

    # Test with quote-like text
    ocr_quote = OcrResult(
        full_text='"The only way to do great work is to love what you do."\n— Steve Jobs',
        lines=[
            '"The only way to do great work is to love what you do."',
            "— Steve Jobs",
        ],
    )
    features_quote = compute_features(ocr_quote)

    assert features_quote.quote_mark_count > 0
    assert features_quote.has_quote_author_pattern is True
    assert features_quote.num_units == 0
    assert features_quote.num_workout_terms == 0


def test_compute_features_edge_cases():
    """Test edge cases in feature computation."""
    # Very short text
    ocr_short = OcrResult(full_text="Hi", lines=["Hi"])
    features = compute_features(ocr_short)
    assert features.layout.line_count == 1
    assert features.num_units == 0

    # Text with no matches
    ocr_none = OcrResult(
        full_text="Random text with no specific patterns here",
        lines=["Random text with no specific patterns here"],
    )
    features = compute_features(ocr_none)
    assert features.has_ingredients_section is False
    assert features.has_steps_section is False
    assert features.has_quote_author_pattern is False

