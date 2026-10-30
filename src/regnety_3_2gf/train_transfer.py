"""
Train model transfer learning trên D-Fire 4 lớp, cùng điều kiện với CustomCNN
và EfficientNet-B0:
  - cùng manifest, augmentation, batch 16 (data.py)
  - cùng hàm train/validate từng epoch (train.py)
  - AdamW, weight_decay 1e-4, CrossEntropyLoss, tối đa 30 epoch,
    early stopping 7, lưu checkpoint theo val_loss
  - Giai đoạn 1 (3 epoch): đóng băng backbone, lr 1e-3
  - Giai đoạn 2: fine-tune toàn mạng, lr 1e-4

Chạy từ thư mục gốc project:
    .venv312\\Scripts\\python.exe src\\train_transfer.py regnet_y_3_2gf
"""

import csv
import sys
import time

import torch
from torch import nn
from torch.optim import AdamW

from data import create_dataloaders
from train import train_one_epoch, validate_one_epoch
from transfer_common import (
    NUM_CLASSES,
    ModelSpec,
    set_backbone_trainable,
    count_parameters,
    fix_manifest_paths,
)


HEAD_EPOCHS = 3
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4

HISTORY_FIELDS = [
    "epoch", "phase", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy", "learning_rate", "elapsed_seconds",
]


def make_optimizer(model, lr):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)


def main(model_key, pretrained=True):
    spec = ModelSpec(model_key)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, _ = create_dataloaders()
    fix_manifest_paths(train_loader, val_loader)
    print(f"Train: {len(train_loader.dataset)} ảnh | "
          f"Val: {len(val_loader.dataset)} ảnh")

    model = spec.build(pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    set_backbone_trainable(model, False, spec.head_name)
    phase, current_lr = "head", LR_HEAD
    optimizer = make_optimizer(model, current_lr)

    print(f"\nModel              : {spec.display_name}")
    print(f"Tổng tham số       : {count_parameters(model):,}")
    print(f"Tham số được train : {count_parameters(model, True):,} "
          f"(giai đoạn 1 - chỉ classifier)")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss, bad_epochs, history = float("inf"), 0, []
    total_start = time.time()

    print(f"\nBắt đầu TRAINING {spec.display_name}...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == HEAD_EPOCHS + 1:
            set_backbone_trainable(model, True, spec.head_name)
            phase, current_lr = "finetune", LR_FINETUNE
            optimizer = make_optimizer(model, current_lr)
            print("\n>>> Mở băng backbone, bắt đầu FINE-TUNE "
                  f"(lr = {current_lr}) | tham số được train: "
                  f"{count_parameters(model, True):,}\n")

        epoch_start = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch, "phase": phase,
            "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
            "learning_rate": current_lr, "elapsed_seconds": elapsed,
        })
        with open(spec.history_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)

        print(f"Epoch {epoch}/{MAX_EPOCHS} [{phase}]")
        print("-" * 50)
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"Val Loss       : {val_loss:.4f}")
        print(f"Val Accuracy   : {val_accuracy:.4f}")
        print(f"Thời gian      : {elapsed:.2f} giây")

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            torch.save({
                "model_name": spec.key,
                "epoch": epoch, "phase": phase,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss, "train_accuracy": train_accuracy,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
                "num_classes": NUM_CLASSES,
                "learning_rate": current_lr, "weight_decay": WEIGHT_DECAY,
            }, spec.best_model_path)
            print(f"Đã lưu best checkpoint: {spec.best_model_path}")
            print("Validation loss đã cải thiện.\n")
        else:
            bad_epochs += 1
            print("Validation loss không cải thiện "
                  f"({bad_epochs}/{EARLY_STOPPING_PATIENCE}).\n")

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping!")
            break

    print("\nTRAINING HOÀN TẤT")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Tổng thời gian train: {(time.time() - total_start) / 60:.1f} phút")
    print(f"Best model: {spec.best_model_path}")
    print(f"History   : {spec.history_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: train_transfer.py regnet_y_3_2gf | efficientnet_b0")
    main(sys.argv[1])
