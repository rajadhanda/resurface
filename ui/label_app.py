"""Streamlit app for labeling images."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_FILE = PROJECT_ROOT / "data" / "labelled" / "labels.csv"

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def load_labels() -> pd.DataFrame:
    """Load or create labels CSV."""
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LABELS_FILE.exists():
        try:
            df = pd.read_csv(LABELS_FILE)
            return df
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return pd.DataFrame(columns=["image_path", "true_label"])
    else:
        # Create empty file with header
        df = pd.DataFrame(columns=["image_path", "true_label"])
        df.to_csv(LABELS_FILE, index=False)
        return df


def save_label(image_path: Path, label: str) -> None:
    """Append a new label to the CSV."""
    df = load_labels()

    # Convert to relative path for storage
    try:
        rel_path = image_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = image_path

    new_row = pd.DataFrame(
        {
            "image_path": [str(rel_path)],
            "true_label": [label],
        }
    )

    # Check if already exists and update, otherwise append
    if "image_path" in df.columns and str(rel_path) in df["image_path"].values:
        df.loc[df["image_path"] == str(rel_path), "true_label"] = label
    else:
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(LABELS_FILE, index=False)
    logger.info(f"Saved label: {rel_path} -> {label}")


def discover_images() -> list[Path]:
    """Discover all images under data/raw/ recursively."""
    images = []
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return images

    for ext in IMAGE_EXTENSIONS:
        images.extend(RAW_DATA_DIR.rglob(f"*{ext}"))

    return sorted(images)


def get_unlabelled_images() -> list[Path]:
    """Get list of images that haven't been labelled yet."""
    all_images = discover_images()
    df = load_labels()

    if df.empty or "image_path" not in df.columns:
        return all_images

    labelled_paths = set(df["image_path"].astype(str))

    unlabelled = []
    for img_path in all_images:
        try:
            rel_path = img_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = img_path

        if str(rel_path) not in labelled_paths:
            unlabelled.append(img_path)

    return unlabelled


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Resurface Labeling App", layout="wide")

    st.title("📸 Resurface Labeling App")
    st.markdown("Label screenshots for the classifier lab")

    # Load labels
    labels_df = load_labels()

    # Sidebar
    with st.sidebar:
        st.header("📊 Summary")

        all_images = discover_images()
        unlabelled = get_unlabelled_images()

        st.metric("Total Images", len(all_images))
        st.metric("Unlabelled", len(unlabelled))
        st.metric("Labelled", len(all_images) - len(unlabelled))

        if not labels_df.empty and "true_label" in labels_df.columns:
            st.subheader("Label Counts")
            label_counts = labels_df["true_label"].value_counts()
            for label, count in label_counts.items():
                st.write(f"**{label}**: {count}")

        # Image selector
        if all_images:
            st.subheader("Jump to Image")
            image_options = [str(img.relative_to(PROJECT_ROOT)) for img in all_images]
            selected_image = st.selectbox(
                "Select image to view/relabel",
                options=[""] + image_options,
            )

            if selected_image:
                selected_path = PROJECT_ROOT / selected_image
                if selected_path.exists():
                    st.session_state["current_image"] = selected_path

    # Main content
    if not unlabelled and not all_images:
        st.warning(
            "No images found. Please add images to the following directories:"
        )
        st.code(f"{RAW_DATA_DIR}/recipes/\n{RAW_DATA_DIR}/workouts/\n{RAW_DATA_DIR}/quotes/\n{RAW_DATA_DIR}/none/")
        return

    # Get current image
    if "current_image" not in st.session_state:
        if unlabelled:
            st.session_state["current_image"] = unlabelled[0]
        elif all_images:
            st.session_state["current_image"] = all_images[0]
        else:
            st.info("All images have been labelled!")
            return

    current_image = st.session_state["current_image"]

    # Check if current image is still unlabelled
    if current_image not in unlabelled and current_image in all_images:
        st.info(f"Image already labelled. Use sidebar to jump to unlabelled images.")

    # Display image
    try:
        st.image(str(current_image), use_container_width=True)
        st.caption(f"**Path:** `{current_image.relative_to(PROJECT_ROOT)}`")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    # Label buttons
    st.subheader("Label this image:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🍳 Recipe", use_container_width=True, type="primary"):
            save_label(current_image, "recipe")
            st.success("Saved as Recipe!")
            st.rerun()

    with col2:
        if st.button("💪 Workout", use_container_width=True, type="primary"):
            save_label(current_image, "workout")
            st.success("Saved as Workout!")
            st.rerun()

    with col3:
        if st.button("💬 Quote", use_container_width=True, type="primary"):
            save_label(current_image, "quote")
            st.success("Saved as Quote!")
            st.rerun()

    with col4:
        if st.button("❌ None", use_container_width=True, type="primary"):
            save_label(current_image, "none")
            st.success("Saved as None!")
            st.rerun()

    with col5:
        if st.button("⏭️ Skip", use_container_width=True):
            # Move to next unlabelled image
            unlabelled = get_unlabelled_images()
            if unlabelled:
                current_idx = unlabelled.index(current_image) if current_image in unlabelled else 0
                next_idx = (current_idx + 1) % len(unlabelled)
                st.session_state["current_image"] = unlabelled[next_idx]
                st.rerun()
            else:
                st.info("No more unlabelled images!")

    # Navigation
    if unlabelled:
        st.markdown("---")
        unlabelled_idx = unlabelled.index(current_image) if current_image in unlabelled else 0
        st.caption(f"Image {unlabelled_idx + 1} of {len(unlabelled)} unlabelled")


if __name__ == "__main__":
    main()

