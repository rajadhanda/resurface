"""Tests for heuristic classifier."""

import pytest
from src.heuristics import (
    score_recipe,
    score_workout,
    score_quote,
    classify,
    ClassificationResult,
)
from src.features import Features, LayoutFeatures
from src.ocr import OcrResult


def create_test_features(
    text: str = "",
    lines: list[str] = None,
    num_units: int = 0,
    num_cooking_verbs: int = 0,
    num_workout_terms: int = 0,
    num_body_parts: int = 0,
    has_ingredients_section: bool = False,
    has_steps_section: bool = False,
    has_quote_author_pattern: bool = False,
    quote_mark_count: int = 0,
    line_count: int = 0,
    bullet_lines: int = 0,
    numbered_lines: int = 0,
    avg_line_length: float = 0.0,
) -> Features:
    """Helper to create test Features objects."""
    if lines is None:
        lines = text.splitlines() if text else []

    ocr = OcrResult(full_text=text, lines=lines)
    layout = LayoutFeatures(
        line_count=line_count or len(lines),
        bullet_lines=bullet_lines,
        numbered_lines=numbered_lines,
        avg_line_length=avg_line_length or (sum(len(l) for l in lines) / len(lines) if lines else 0.0),
    )

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


def test_score_recipe():
    """Test recipe scoring."""
    # High recipe score
    feats_recipe = create_test_features(
        text="Ingredients\n2 cups flour\n1 tbsp sugar\nMix and bake",
        lines=["Ingredients", "2 cups flour", "1 tbsp sugar", "Mix and bake"],
        num_units=3,
        num_cooking_verbs=2,
        has_ingredients_section=True,
        bullet_lines=3,
        numbered_lines=2,
    )
    score = score_recipe(feats_recipe)
    assert score >= 5.0  # Should score well

    # Low recipe score
    feats_low = create_test_features(
        text="Random text",
        num_units=0,
        num_cooking_verbs=0,
        has_ingredients_section=False,
    )
    score_low = score_recipe(feats_low)
    assert score_low < score


def test_score_workout():
    """Test workout scoring."""
    # High workout score
    feats_workout = create_test_features(
        text="3 sets of 10 reps\nRest 60 seconds\nLegs workout",
        lines=["3 sets of 10 reps", "Rest 60 seconds", "Legs workout"],
        num_workout_terms=3,
        num_body_parts=1,
        bullet_lines=2,
    )
    score = score_workout(feats_workout)
    assert score >= 5.0  # Should score well

    # Workout with ingredients section (should subtract)
    feats_mixed = create_test_features(
        text="Ingredients\n3 sets of 10 reps",
        has_ingredients_section=True,
        num_workout_terms=2,
    )
    score_mixed = score_workout(feats_mixed)
    assert score_mixed < score  # Should be lower due to penalty


def test_score_quote():
    """Test quote scoring."""
    # High quote score
    feats_quote = create_test_features(
        text='"The only way to do great work is to love what you do."\n— Steve Jobs',
        lines=[
            '"The only way to do great work is to love what you do."',
            "— Steve Jobs",
        ],
        has_quote_author_pattern=True,
        quote_mark_count=2,
        line_count=2,
        avg_line_length=50.0,
        num_units=0,
        num_workout_terms=0,
    )
    score = score_quote(feats_quote)
    assert score >= 5.0  # Should score well

    # Low quote score
    feats_low = create_test_features(
        text="Random text with no quotes",
        quote_mark_count=0,
        has_quote_author_pattern=False,
        num_units=5,  # Has units, not a quote
    )
    score_low = score_quote(feats_low)
    assert score_low < score


def test_classify():
    """Test classification function."""
    # Recipe classification
    feats_recipe = create_test_features(
        text="Ingredients\n2 cups flour\nMix and bake",
        lines=["Ingredients", "2 cups flour", "Mix and bake"],
        num_units=3,
        num_cooking_verbs=2,
        has_ingredients_section=True,
    )
    result = classify(feats_recipe, threshold=5.0)
    assert isinstance(result, ClassificationResult)
    assert "recipe" in result.scores
    assert "workout" in result.scores
    assert "quote" in result.scores
    assert result.scores["recipe"] >= result.scores["workout"]
    assert result.scores["recipe"] >= result.scores["quote"]
    if result.scores["recipe"] >= 5.0:
        assert result.item_type == "recipe"

    # Workout classification
    feats_workout = create_test_features(
        text="3 sets of 10 reps\nRest 60 seconds",
        lines=["3 sets of 10 reps", "Rest 60 seconds"],
        num_workout_terms=3,
        num_body_parts=1,
    )
    result = classify(feats_workout, threshold=5.0)
    if result.scores["workout"] >= 5.0:
        assert result.item_type == "workout"

    # Quote classification
    feats_quote = create_test_features(
        text='"Great quote here"\n— Author Name',
        lines=['"Great quote here"', "— Author Name"],
        has_quote_author_pattern=True,
        quote_mark_count=2,
        line_count=2,
        avg_line_length=30.0,
        num_units=0,
        num_workout_terms=0,
    )
    result = classify(feats_quote, threshold=5.0)
    if result.scores["quote"] >= 5.0:
        assert result.item_type == "quote"

    # None classification (all scores below threshold)
    feats_none = create_test_features(
        text="Random text with no clear pattern",
        num_units=0,
        num_cooking_verbs=0,
        num_workout_terms=0,
        has_ingredients_section=False,
    )
    result = classify(feats_none, threshold=10.0)  # High threshold
    assert result.item_type == "none"


def test_classify_threshold():
    """Test that threshold works correctly."""
    feats_recipe = create_test_features(
        text="Ingredients\n2 cups flour\nMix and bake",
        lines=["Ingredients", "2 cups flour", "Mix and bake"],
        num_units=3,
        num_cooking_verbs=2,
        has_ingredients_section=True,
    )

    # Low threshold - should classify
    result_low = classify(feats_recipe, threshold=1.0)
    assert result_low.item_type != "none"

    # High threshold - might classify as none
    result_high = classify(feats_recipe, threshold=100.0)
    assert result_high.item_type == "none"

