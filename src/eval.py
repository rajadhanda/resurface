"""Evaluation module for classifier metrics."""

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .features import compute_features
from .heuristics import ClassificationResult, ItemType, classify
from .ocr import run_ocr

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate_dataset(threshold: float = 5.0) -> Dict:
    """
    Load labelled dataset, run classifier, return metrics.

    Args:
        threshold: Classification threshold (default: 5.0)

    Returns:
        Dictionary with:
        - confusion_matrix: 4x4 numpy array (recipe, workout, quote, none)
        - per_class_precision: dict
        - per_class_recall: dict
        - overall_accuracy: float
        - classification_report: str
    """
    # Load labels CSV
    labels_file = Path(__file__).parent.parent / "data" / "labelled" / "labels.csv"
    labels_file.parent.mkdir(parents=True, exist_ok=True)

    if not labels_file.exists():
        logger.warning(f"Labels file not found: {labels_file}. Creating empty file.")
        labels_file.write_text("image_path,true_label\n", encoding="utf-8")
        return {
            "confusion_matrix": np.zeros((4, 4), dtype=int),
            "per_class_precision": {},
            "per_class_recall": {},
            "overall_accuracy": 0.0,
            "classification_report": "No data available",
            "message": "No labelled data found. Please label some images first.",
        }

    try:
        df = pd.read_csv(labels_file)
    except Exception as e:
        logger.error(f"Failed to load labels file: {e}")
        return {
            "confusion_matrix": np.zeros((4, 4), dtype=int),
            "per_class_precision": {},
            "per_class_recall": {},
            "overall_accuracy": 0.0,
            "classification_report": f"Error loading labels: {e}",
        }

    if df.empty or "image_path" not in df.columns or "true_label" not in df.columns:
        logger.warning("Labels file is empty or missing required columns.")
        return {
            "confusion_matrix": np.zeros((4, 4), dtype=int),
            "per_class_precision": {},
            "per_class_recall": {},
            "overall_accuracy": 0.0,
            "classification_report": "No valid data in labels file",
        }

    # Process each image
    true_labels: list[ItemType] = []
    predicted_labels: list[ItemType] = []

    for _, row in df.iterrows():
        image_path_str = str(row["image_path"])
        true_label_str = str(row["true_label"]).lower().strip()

        # Validate true label
        if true_label_str not in ["recipe", "workout", "quote", "none"]:
            logger.warning(f"Invalid label '{true_label_str}' for {image_path_str}, skipping")
            continue

        true_label: ItemType = true_label_str  # type: ignore

        # Resolve image path (could be relative or absolute)
        image_path = Path(image_path_str)
        if not image_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            image_path = project_root / image_path

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}, skipping")
            continue

        try:
            # Run OCR
            ocr_result = run_ocr(image_path)

            # Compute features
            features = compute_features(ocr_result)

            # Classify
            result: ClassificationResult = classify(features, threshold=threshold)

            true_labels.append(true_label)
            predicted_labels.append(result.item_type)

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

    if not true_labels:
        return {
            "confusion_matrix": np.zeros((4, 4), dtype=int),
            "per_class_precision": {},
            "per_class_recall": {},
            "overall_accuracy": 0.0,
            "classification_report": "No valid images processed",
        }

    # Compute metrics
    label_order = ["recipe", "workout", "quote", "none"]
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_order,  # type: ignore
    )

    # Calculate precision and recall per class
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_order,  # type: ignore
        output_dict=True,
        zero_division=0,
    )

    per_class_precision = {
        label: report[label]["precision"] for label in label_order if label in report
    }
    per_class_recall = {
        label: report[label]["recall"] for label in label_order if label in report
    }

    overall_accuracy = accuracy_score(true_labels, predicted_labels)

    return {
        "confusion_matrix": cm,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "overall_accuracy": overall_accuracy,
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=label_order,  # type: ignore
        ),
        "num_samples": len(true_labels),
    }


if __name__ == "__main__":
    from pprint import pprint

    results = evaluate_dataset(threshold=5.0)
    pprint(results)

