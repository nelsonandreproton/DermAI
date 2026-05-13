"""
EDA: show class distribution and sample images.
Run after download_datasets.py.
"""

from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "prepared"


def count_images(split: str) -> Counter:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return Counter()
    counts = Counter()
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*.jpg")))
    return counts


def main():
    for split in ["train", "val"]:
        counts = count_images(split)
        if not counts:
            print(f"{split}: no data found")
            continue
        total = sum(counts.values())
        print(f"\n=== {split} ({total} images) ===")
        for cls, n in sorted(counts.items()):
            bar = "█" * (n // 10)
            print(f"  {cls:<12} {n:>4}  {bar}")

    print("\nIf any class < 50 images, consider adding more data or removing that class.")


if __name__ == "__main__":
    main()
