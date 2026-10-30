"""
Train EfficientNet-B0 (transfer learning) trên D-Fire 4 lớp.

Giữ nguyên điều kiện so sánh với CustomCNN:
  - cùng manifest train/val, cùng augmentation, batch size 16 (data.py)
  - cùng hàm train/validate từng epoch (train.py)
  - AdamW, weight_decay 1e-4, CrossEntropyLoss, tối đa 30 epoch,
    early stopping patience 7, lưu checkpoint theo val_loss

Khác biệt (đặc trưng của transfer learning):
  - Giai đoạn 1: đóng băng backbone, chỉ train lớp phân loại (lr 1e-3)
  - Giai đoạn 2: mở băng toàn bộ, fine-tune với lr nhỏ (1e-4)

Chạy từ thư mục gốc project:
    .venv312\\Scripts\\python src\\train_efficientnet.py
"""

import csv
import time

import torch
from torch import nn
from torch.optim import AdamW

from data import create_dataloaders
from train import train_one_epoch, validate_one_epoch
from efficientnet_common import (
    MODEL_NAME,
    NUM_CLASSES,
    OUTPUT_DIR,
    BEST_MODEL_PATH,
    HISTORY_PATH,
    build_efficientnet_b0,
    set_backbone_trainable,
    count_parameters,
    fix_manifest_paths,
)


HEAD_EPOCHS = 3                 # giai đoạn 1: chỉ train classifier
MAX_EPOCHS = 30                 # tổng số epoch tối đa (giống CustomCNN)
EARLY_STOPPING_PATIENCE = 7

LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4

HISTORY_FIELDS = [
    "epoch",
    "phase",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "learning_rate",
    "elapsed_seconds",
]


def save_history(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def save_best_checkpoint(model, optimizer, epoch, phase,
                         train_loss, train_accuracy,
                         val_loss, val_accuracy, learning_rate):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": MODEL_NAME,
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num_classes": NUM_CLASSES,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
    }
    torch.save(checkpoint, BEST_MODEL_PATH)
    print(f"Đã lưu best checkpoint: {BEST_MODEL_PATH}")


def make_optimizer(model, lr):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, _ = create_dataloaders()
    fix_manifest_paths(train_loader, val_loader)
    print(f"Train: {len(train_loader.dataset)} ảnh | "
          f"Val: {len(val_loader.dataset)} ảnh")

    model = build_efficientnet_b0().to(device)
    criterion = nn.CrossEntropyLoss()

    # ---------- Giai đoạn 1: đóng băng backbone ----------
    set_backbone_trainable(model, False)
    phase = "head"
    current_lr = LR_HEAD
    optimizer = make_optimizer(model, current_lr)

    print(f"\nTổng tham số       : {count_parameters(model):,}")
    print(f"Tham số được train : {count_parameters(model, True):,} "
          f"(giai đoạn 1 - chỉ classifier)")

    best_val_loss = float("inf")
    bad_epochs = 0
    history = []
    total_start = time.time()

    print("\nBắt đầu TRAINING EfficientNet-B0...\n")

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---------- Chuyển sang giai đoạn 2: fine-tune ----------
        if epoch == HEAD_EPOCHS + 1:
            set_backbone_trainable(model, True)
            phase = "finetune"
            current_lr = LR_FINETUNE
            optimizer = make_optimizer(model, current_lr)
            print("\n>>> Mở băng backbone, bắt đầu FINE-TUNE "
                  f"(lr = {current_lr}) | tham số được train: "
                  f"{count_parameters(model, True):,}\n")

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "elapsed_seconds": elapsed,
        })
        save_history(history)

        print(f"Epoch {epoch}/{MAX_EPOCHS} [{phase}]")
        print("-" * 50)
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"Val Loss       : {val_loss:.4f}")
        print(f"Val Accuracy   : {val_accuracy:.4f}")
        print(f"Thời gian      : {elapsed:.2f} giây")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_best_checkpoint(
                model, optimizer, epoch, phase,
                train_loss, train_accuracy,
                val_loss, val_accuracy, current_lr,
            )
            print("Validation loss đã cải thiện.\n")
        else:
            bad_epochs += 1
            print("Validation loss không cải thiện "
                  f"({bad_epochs}/{EARLY_STOPPING_PATIENCE}).\n")

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping!")
            break

    total_minutes = (time.time() - total_start) / 60

    print("\nTRAINING HOÀN TẤT")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Tổng thời gian train: {total_minutes:.1f} phút")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"History   : {HISTORY_PATH}")


if __name__ == "__main__":
    main()
