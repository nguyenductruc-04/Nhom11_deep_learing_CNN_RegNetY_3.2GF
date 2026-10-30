from pathlib import Path
import json

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data import create_dataloaders
from model import CustomCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pt"

FIGURE_DIR = PROJECT_ROOT / "figures"

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "test_metrics.json"
)

CONFUSION_MATRIX_PATH = (
    FIGURE_DIR
    / "confusion_matrix.png"
)


CLASS_NAMES = [
    "none",
    "smoke",
    "fire",
    "smoke_fire",
]

NUM_CLASSES = 4


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

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

        predictions = logits.argmax(
            dim=1
        )

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

    return y_true, y_pred


def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nTEST METRICS")
    print("-" * 50)

    print(
        f"Accuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.4f}"
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 50)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(report)

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("CONFUSION MATRIX")
    print("-" * 50)

    print(matrix)


def save_results(y_true, y_pred):
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    ax.figure.colorbar(
        image,
        ax=ax
    )

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - D-Fire CNN"
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200
    )

    plt.close()

    print(
        f"\nĐã lưu metrics: "
        f"{METRICS_PATH}"
    )

    print(
        f"Đã lưu confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
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

    _, _, test_loader = (
        create_dataloaders()
    )

    model = CustomCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded best checkpoint "
        f"from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    print(
        f"Best val accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )

    y_true, y_pred = evaluate_test(
        model,
        test_loader,
        device
    )

    print_metrics(
        y_true,
        y_pred
    )

    save_results(
        y_true,
        y_pred
    )


if __name__ == "__main__":
    main()