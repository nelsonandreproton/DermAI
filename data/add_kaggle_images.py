"""
Add images downloaded manually from Kaggle into the prepared dataset.

Instructions:
1. Download from Kaggle manually (no API key needed):
   - https://www.kaggle.com/datasets/shubhamgoel27/dermnet  (23 classes, includes psoriasis)
   - https://www.kaggle.com/datasets/pallapurajkumar/psoriasis-skin-dataset

2. Extract ZIP to any folder (e.g. C:/kaggle_skin/)

3. Run this script pointing at the extracted folder:
   python data/add_kaggle_images.py --src "C:/kaggle_skin/dermnet/train"

The script copies images into data/prepared/train|val, preserving existing images.
"""

import argparse
import random
from pathlib import Path
from PIL import Image

OUT_DIR = Path(__file__).parent / "prepared"
IMG_SIZE = 224
VAL_RATIO = 0.2  # 20% of new images go to val

LABEL_MAP = {
    "psoria": "psoriasis",
    # lichen planus excluded — resembles eczema plaques, contaminates psoriasis class
    "eczema": "eczema",
    "atopic": "eczema",
    "dermatit": "eczema",
    "acne": "acne",
    # rosacea excluded — visually distinct from acne papules
    # DermNet "Melanoma Skin Cancer Nevi and Moles" excluded entirely — mixed folder,
    # cannot auto-split; use AICamp's pure melanoma folder instead
    # actinic/basal cell excluded — known regression (not true melanoma)
    "nevi": "nevi",
    "nevus": "nevi",
    "mole": "nevi",
    "seborrheic": "nevi",
}


EXCLUDE_FOLDERS = {
    # mixed melanoma+nevi content — cannot auto-label
    "melanoma skin cancer nevi and moles",
}


def normalize_label(folder_name: str) -> str | None:
    name = folder_name.lower()
    if name in EXCLUDE_FOLDERS:
        return None
    for key, label in LABEL_MAP.items():
        if key in name:
            return label
    return None


def count_existing(split: str, label: str) -> int:
    d = OUT_DIR / split / label
    if not d.exists():
        return 0
    return len(list(d.glob("*.jpg")))


def next_idx(split: str, label: str) -> int:
    return count_existing(split, label)


def process_folder(src_dir: Path) -> dict:
    added = {"train": {}, "val": {}}

    class_folders = [d for d in src_dir.iterdir() if d.is_dir()]
    print(f"Found {len(class_folders)} folders in {src_dir}\n")

    for class_folder in class_folders:
        label = normalize_label(class_folder.name)
        if label is None:
            print(f"  Skip (unknown): {class_folder.name}")
            continue

        img_files = (
            list(class_folder.glob("*.jpg"))
            + list(class_folder.glob("*.jpeg"))
            + list(class_folder.glob("*.png"))
        )
        random.shuffle(img_files)

        val_count = max(1, int(len(img_files) * VAL_RATIO))
        val_files = set(img_files[:val_count])
        train_files = img_files[val_count:]

        for split, files in [("train", train_files), ("val", list(val_files))]:
            out_dir = OUT_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = next_idx(split, label)
            count = 0
            for img_path in files:
                try:
                    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    img.save(out_dir / f"{label}_{idx:04d}.jpg")
                    idx += 1
                    count += 1
                except Exception as e:
                    print(f"    Skip {img_path.name}: {e}")
            added[split][label] = added[split].get(label, 0) + count

        print(f"  {label:<12} <- {class_folder.name} ({len(train_files)} train, {val_count} val)")

    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to extracted Kaggle dataset folder")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: {src} does not exist")
        return

    print(f"Adding images from: {src}\n")
    added = process_folder(src)

    print(f"\nAdded:")
    print(f"  train: {added['train']}")
    print(f"  val:   {added['val']}")
    print("\nRun: python data/explore.py to see updated counts")
    print("Then: python train/train.py to retrain")


if __name__ == "__main__":
    main()
