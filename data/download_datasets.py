"""
Download skin lesion dataset from Hugging Face using direct file download.
Avoids the datasets library which hangs on Windows with long paths.
Classes: psoriasis, eczema, acne, melanoma, nevi
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, login
from PIL import Image
from dotenv import load_dotenv

CLASSES = ["psoriasis", "eczema", "acne", "melanoma", "nevi"]
OUT_DIR = Path(__file__).parent / "prepared"
MAX_PER_CLASS = 9999  # use all available images

# Map dataset folder name substrings → our class labels
# Keys are substrings matched against lowercased folder name
LABEL_MAP = {
    "psoria": "psoriasis",
    "lichen planus": "psoriasis",   # same disease family, visually similar plaques
    "eczema": "eczema",
    "atopic": "eczema",
    "contact dermatitis": "eczema",
    "acne": "acne",
    "rosacea": "acne",
    "melanoma": "melanoma",
    # actinic/basal cell excluded — visually distinct from melanoma, confuses model
    "nevi": "nevi",
    "moles": "nevi",
    "seborrheic": "nevi",
    "benign tumor": "nevi",
}


def normalize_label(folder_name: str) -> str | None:
    name = folder_name.lower()
    for key, label in LABEL_MAP.items():
        if key in name:
            return label
    return None


def process_split(raw_dir: Path, split_name: str) -> dict:
    split_dir = OUT_DIR / split_name
    counts = {c: 0 for c in CLASSES}

    # raw_dir should contain class subfolders
    if not raw_dir.exists():
        print(f"  Split dir not found: {raw_dir}")
        return counts

    class_folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"  Found {len(class_folders)} class folders in {split_name}")

    for class_folder in class_folders:
        label = normalize_label(class_folder.name)
        if label is None:
            print(f"  Skipping unknown class: {class_folder.name}")
            continue

        out_class_dir = split_dir / label
        out_class_dir.mkdir(parents=True, exist_ok=True)

        img_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))

        for img_path in img_files:
            if counts[label] >= MAX_PER_CLASS:
                break
            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                idx = counts[label]
                img.save(out_class_dir / f"{label}_{idx:04d}.jpg")
                counts[label] += 1
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

    return counts


def main():
    # load .env from project root
    load_dotenv(Path(__file__).parent.parent / ".env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use short local path to avoid Windows 260-char limit
    cache_dir = Path("C:/hf_cache")
    cache_dir.mkdir(exist_ok=True)

    # Auth: read token from env or prompt
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("Authenticated with HF token.\n")
    else:
        print("No HF_TOKEN found in environment. Proceeding unauthenticated (may be rate limited).\n")

    print("Downloading dataset snapshot (this may take a few minutes)...")
    print("Using cache: C:/hf_cache\n")

    local_dir = snapshot_download(
        repo_id="notable12/AICamp-2023-Skin-Conditions-Dataset",
        repo_type="dataset",
        local_dir=str(cache_dir / "aicamp-skin"),
        token=hf_token,
    )

    print(f"\nDownloaded to: {local_dir}")
    print("Processing images...\n")

    local_path = Path(local_dir)

    # Images are under "Unzipped ProcessedDataset/train" and "test"
    unzipped = local_path / "Unzipped ProcessedDataset"
    train_dir = unzipped / "train"
    val_dir = unzipped / "test"

    if not train_dir.exists():
        print(f"ERROR: train dir not found at {train_dir}")
        print("Top-level contents:")
        for p in local_path.iterdir():
            print(f"  {p}")
        return

    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}\n")

    train_counts = process_split(train_dir, "train")
    val_counts = process_split(val_dir, "val")

    print(f"\ntrain: {train_counts}")
    print(f"val:   {val_counts}")

    print("\nDone. Run: python data/explore.py")


if __name__ == "__main__":
    main()
