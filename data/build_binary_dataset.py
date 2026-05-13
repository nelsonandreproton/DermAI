"""
Build binary dataset: psoriasis vs not_psoriasis.
Sources:
  - AICamp HF dataset (already cached at C:/hf_cache/aicamp-skin)
  - DermNet Kaggle (C:/kaggle_skin/dermnet)
"""

import random
import shutil
from pathlib import Path
from PIL import Image

OUT_DIR = Path(__file__).parent / "prepared"
IMG_SIZE = 224
VAL_RATIO = 0.2
SEED = 42

# AICamp folders → binary label
AICAMP_MAP = {
    "psoriasis": "psoriasis",
    "lichen planus": "psoriasis",
    "eczema": "not_psoriasis",
    "atopic dermatitis": "not_psoriasis",
    "contact dermatitis": "not_psoriasis",
    "acne": "not_psoriasis",
    "rosacea": "not_psoriasis",
    "melanoma": "not_psoriasis",
    "nevi": "not_psoriasis",
    "moles": "not_psoriasis",
    "seborrheic": "not_psoriasis",
    "benign tumor": "not_psoriasis",
}

# DermNet folders → binary label (exact lowercased folder name)
DERMNET_MAP = {
    "psoriasis pictures lichen planus and related diseases": "psoriasis",
    "eczema photos": "not_psoriasis",
    "atopic dermatitis photos": "not_psoriasis",
    "acne and rosacea photos": "not_psoriasis",
    "melanoma skin cancer nevi and moles": "not_psoriasis",
    "seborrheic keratoses and other benign tumors": "not_psoriasis",
    "poison ivy photos and other contact dermatitis": "not_psoriasis",
    "bullous disease photos": "not_psoriasis",
    "tinea ringworm candidiasis and other fungal infections": "not_psoriasis",
    "warts molluscum and other viral infections": "not_psoriasis",
    "actinic keratosis basal cell carcinoma and other malignant lesions": "not_psoriasis",
}


def save_images(img_files: list, split: str, label: str, start_idx: int) -> int:
    out_dir = OUT_DIR / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in img_files:
        try:
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img.save(out_dir / f"{label}_{start_idx + count:05d}.jpg")
            count += 1
        except Exception as e:
            print(f"    Skip {img_path.name}: {e}")
    return count


def add_folder(src: Path, label: str, counters: dict) -> None:
    img_files = (
        list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png"))
    )
    if not img_files:
        return

    random.shuffle(img_files)
    n_val = max(1, int(len(img_files) * VAL_RATIO))
    val_files = img_files[:n_val]
    train_files = img_files[n_val:]

    added_train = save_images(train_files, "train", label, counters[f"{label}_train"])
    added_val = save_images(val_files, "val", label, counters[f"{label}_val"])
    counters[f"{label}_train"] += added_train
    counters[f"{label}_val"] += added_val
    print(f"  {label:<16} <- {src.name} (+{added_train} train, +{added_val} val)")


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counters = {
        "psoriasis_train": 0, "psoriasis_val": 0,
        "not_psoriasis_train": 0, "not_psoriasis_val": 0,
    }

    # --- AICamp ---
    aicamp = Path("C:/hf_cache/aicamp-skin/Unzipped ProcessedDataset")
    print("\n=== AICamp ===")
    for split_name, split_folder in [("train", aicamp / "train"), ("test", aicamp / "test")]:
        if not split_folder.exists():
            print(f"  Not found: {split_folder}")
            continue
        for cls_folder in sorted(split_folder.iterdir()):
            if not cls_folder.is_dir():
                continue
            name = cls_folder.name.lower()
            label = None
            for key, lbl in AICAMP_MAP.items():
                if key in name:
                    label = lbl
                    break
            if label is None:
                print(f"  Skip: {cls_folder.name}")
                continue
            add_folder(cls_folder, label, counters)

    # --- DermNet ---
    dermnet = Path("C:/kaggle_skin/dermnet/train")
    print("\n=== DermNet ===")
    if dermnet.exists():
        for cls_folder in sorted(dermnet.iterdir()):
            if not cls_folder.is_dir():
                continue
            name = cls_folder.name.lower()
            label = DERMNET_MAP.get(name)
            if label is None:
                print(f"  Skip: {cls_folder.name}")
                continue
            add_folder(cls_folder, label, counters)
    else:
        print(f"  Not found: {dermnet}")

    print("\n=== Result ===")
    for split in ["train", "val"]:
        total = counters[f"psoriasis_{split}"] + counters[f"not_psoriasis_{split}"]
        print(f"  {split}: psoriasis={counters[f'psoriasis_{split}']}  not_psoriasis={counters[f'not_psoriasis_{split}']}  total={total}")

    print("\nDone. Run: python train/train.py")


if __name__ == "__main__":
    main()
