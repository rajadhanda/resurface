# Resurface Classifier Lab

A lab for building and evaluating a Phase 0 heuristic classifier for screenshots (recipes, workouts, quotes, none).

## Overview

This project provides a self-contained Python environment to:
1. Load screenshots (recipes, workouts, quotes, random stuff)
2. Label them via a simple web UI (Streamlit)
3. Run a **Phase 0 heuristic classifier** on OCR text + layout features
4. Evaluate the classifier with real metrics (confusion matrix, precision/recall) and tune thresholds
5. Later, reuse the logic in an iOS/Android app (core functions are clean and portable)

## Setup

### Prerequisites

- Python 3.11 or higher
- Tesseract OCR installed on your system

### Installing Tesseract

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki) or use:
```bash
choco install tesseract
```

### Project Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Add Images

Drop your screenshots into the appropriate directories:
- `data/raw/recipes/` - Recipe screenshots
- `data/raw/workouts/` - Workout screenshots
- `data/raw/quotes/` - Quote screenshots
- `data/raw/none/` - Other screenshots

### 2. Label Images

Run the Streamlit labeling app:
```bash
streamlit run ui/label_app.py
```

The app will:
- Show unlabelled images one at a time
- Let you label them as Recipe, Workout, Quote, or None
- Save labels to `data/labelled/labels.csv`

### 3. Evaluate Classifier

Run the evaluation script to see metrics:
```bash
python -m src.eval
```

This will:
- Load all labelled images
- Run OCR (with caching)
- Compute features
- Classify using heuristics
- Print confusion matrix, precision/recall, and overall accuracy

### 4. Tune Thresholds

Use the Jupyter notebooks to explore and tune:
- `notebooks/01_explore_dataset.ipynb` - Explore the dataset
- `notebooks/02_tune_thresholds.ipynb` - Sweep thresholds and plot results
- `notebooks/03_future_ml_model.ipynb` - Placeholder for ML model work

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw screenshots (recipes/, workouts/, quotes/, none/)
│   ├── labelled/         # labels.csv with ground truth
│   └── ocr_cache/        # Cached OCR results (auto-generated)
├── src/
│   ├── ocr.py           # OCR with caching
│   ├── features.py      # Feature extraction
│   ├── heuristics.py    # Phase 0 classifier
│   └── eval.py          # Evaluation metrics
├── ui/
│   └── label_app.py     # Streamlit labeling interface
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
└── requirements.txt    # Python dependencies
```

## How the Classifier Works

The **Phase 0 heuristic classifier** uses rule-based scoring:

- **Recipe score**: Based on ingredient sections, cooking verbs, measurement units, bullet/numbered lists
- **Workout score**: Based on sets/reps patterns, workout terms, body parts, exercise notation (e.g., 3x10)
- **Quote score**: Based on quote marks, author patterns, line count, prose-like text

The classifier picks the highest-scoring type if it exceeds a threshold (default: 5.0), otherwise returns "none".

## Next Steps

1. **Collect more data**: Label more images to improve evaluation
2. **Tune thresholds**: Use `notebooks/02_tune_thresholds.ipynb` to find optimal thresholds
3. **Train ML model**: Once you have enough data, train a small ML model (see `notebooks/03_future_ml_model.ipynb`)
4. **Deploy to mobile**: The core functions in `src/` are designed to be portable for iOS/Android apps

## Running Tests

```bash
pytest
```

## Notes

- OCR results are cached in `data/ocr_cache/` to avoid re-processing the same images
- The classifier is designed to be easy to tweak - edit scoring functions in `src/heuristics.py`
- All core logic is in `src/` and can be reused in mobile apps

## License

MIT
