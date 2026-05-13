"""
Train a skin lesion classifier.
Backbone: EfficientNet-B0 (better accuracy than MobileNetV3 on skin datasets).
Two-phase strategy:
  Phase 1 (epochs 1-15): frozen backbone, train head only (LR=1e-3)
  Phase 2 (epochs 16-30): unfreeze last 3 backbone blocks, low LR (1e-4)
"""

import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

DATA_DIR = Path(__file__).parent.parent / "data" / "prepared"
MODEL_DIR = Path(__file__).parent.parent / "model"

EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 15
BATCH_SIZE = 32
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
IMG_SIZE = 224


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # freeze entire backbone for phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # replace classifier head
    # EfficientNet-B0 classifier: [Dropout, Linear(1280, 1000)]
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 3) -> None:
    # EfficientNet-B0 features: [Conv, MBConv x7, Conv] = 9 children (indices 0-8)
    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfroze last {n_blocks} blocks. Trainable params: {trainable:,}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=get_transforms(True))
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=get_transforms(False))

    if len(train_ds) == 0:
        print("No training data found. Run data/download_datasets.py first.")
        return

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_counts = [0] * num_classes
    for _, label in train_ds.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    weights = torch.tensor([total / (num_classes * c) for c in class_counts])
    print(f"Class weights: { {class_names[i]: f'{weights[i]:.2f}' for i in range(num_classes)} }")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    MODEL_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    # --- Phase 1: frozen backbone, train head only ---
    print(f"\n=== Phase 1: frozen backbone ({EPOCHS_PHASE1} epochs) ===")
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS_PHASE1} | "
            f"train={train_acc:.2%} val={val_acc:.2%} | loss={val_loss:.3f} | {elapsed:.0f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"  -> saved (val_acc={val_acc:.2%})")

    # --- Phase 2: unfreeze last 3 backbone blocks, low LR ---
    print(f"\n=== Phase 2: partial unfreeze ({EPOCHS_PHASE2} epochs, LR={LR_PHASE2}) ===")
    unfreeze_last_blocks(model, n_blocks=3)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS_PHASE2} | "
            f"train={train_acc:.2%} val={val_acc:.2%} | loss={val_loss:.3f} | {elapsed:.0f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"  -> saved (val_acc={val_acc:.2%})")

    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print(f"\nBest val accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {MODEL_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
