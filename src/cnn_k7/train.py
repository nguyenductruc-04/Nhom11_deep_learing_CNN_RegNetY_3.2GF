import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from data import create_dataloaders
from model import CustomCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"


NUM_CLASSES = 4

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


def save_best_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "num_classes":
            NUM_CLASSES,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )

    print(
        f"Đã lưu best checkpoint: "
        f"{BEST_MODEL_PATH}"
    )


def save_history(history):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "elapsed_seconds",
    ]

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            history
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_loader, val_loader, _ = (
        create_dataloaders()
    )

    model = CustomCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    bad_epochs = 0

    history = []

    print(
        "\nBắt đầu FULL TRAINING...\n"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):
        epoch_start_time = (
            time.time()
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_accuracy = (
            validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        elapsed_seconds = (
            time.time()
            - epoch_start_time
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "learning_rate":
                    current_lr,

                "elapsed_seconds":
                    elapsed_seconds,
            }
        )

        save_history(
            history
        )

        print(
            f"\nEpoch "
            f"{epoch}/{MAX_EPOCHS}"
        )

        print(
            "-" * 50
        )

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Learning Rate  : "
            f"{current_lr:.6f}"
        )

        print(
            f"Thời gian      : "
            f"{elapsed_seconds:.2f} giây"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            bad_epochs = 0

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            print(
                "Validation loss "
                "đã cải thiện."
            )

        else:
            bad_epochs += 1

            print(
                "Validation loss "
                "không cải thiện."
            )

            print(
                "Số epoch không "
                f"cải thiện: "
                f"{bad_epochs}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            bad_epochs
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping!"
            )

            break

    print(
        "\nFULL TRAINING HOÀN TẤT"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"History: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()