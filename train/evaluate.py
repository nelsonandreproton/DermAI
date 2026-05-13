"""
Evaluate trained model: confusion matrix + per-class F1.
Run after train.py.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

MODEL_DIR = Path(__file__).parent.parent / "model"
DATA_DIR = Path(__file__).parent.parent / "data" / "prepared"
IMG_SIZE = 224


def load_model(num_classes: int):
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location="cpu"))
    model.eval()
    return model


def confusion_matrix(preds, labels, num_classes):
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    return matrix


def main():
    with open(MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    model = load_model(num_classes)

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs)
            preds = out.argmax(1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f"Overall accuracy: {correct/len(all_labels):.2%} ({correct}/{len(all_labels)})\n")

    # per-class F1
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 42)
    matrix = confusion_matrix(all_preds, all_labels, num_classes)
    for i, cls in enumerate(class_names):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(num_classes)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"{cls:<14} {precision:>10.2%} {recall:>8.2%} {f1:>6.2%}")

    # confusion matrix
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    header = f"{'':14}" + "".join(f"{c[:6]:>8}" for c in class_names)
    print(header)
    for i, cls in enumerate(class_names):
        row = f"{cls:<14}" + "".join(f"{matrix[i][j]:>8}" for j in range(num_classes))
        print(row)


if __name__ == "__main__":
    main()
