"""OCR module with caching support."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OcrResult:
    """OCR result containing full text and line-by-line breakdown."""

    full_text: str
    lines: List[str]


def run_ocr(image_path: Path, cache_dir: Path = None) -> OcrResult:
    """
    Run OCR on an image and return text + lines.

    Use a simple JSON cache in data/ocr_cache so we don't repeatedly OCR the same file.

    Args:
        image_path: Path to the image file
        cache_dir: Directory for caching OCR results (defaults to data/ocr_cache)

    Returns:
        OcrResult with full_text and lines
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "ocr_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache file path based on image basename
    cache_file = cache_dir / f"{image_path.stem}.json"

    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.info(f"Loaded OCR result from cache: {cache_file}")
                return OcrResult(
                    full_text=cached_data["full_text"],
                    lines=cached_data["lines"],
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}. Re-running OCR.")

    # Run OCR
    try:
        logger.info(f"Running OCR on {image_path}")
        image = Image.open(image_path)
        full_text = pytesseract.image_to_string(image).strip()

        # Split into lines and normalize
        lines = [
            line.strip()
            for line in full_text.splitlines()
            if line.strip()  # Remove empty lines
        ]

        result = OcrResult(full_text=full_text, lines=lines)

        # Cache the result
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Cached OCR result to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache OCR result: {e}")

        return result

    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        # Return empty result on failure
        return OcrResult(full_text="", lines=[])

