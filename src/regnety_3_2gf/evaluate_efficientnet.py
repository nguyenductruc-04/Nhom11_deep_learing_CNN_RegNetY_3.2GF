"""
Đánh giá EfficientNet-B0 trên tập test và so sánh với CustomCNN.

Đầu ra:
  outputs/efficientnet_b0/test_metrics.json
  outputs/efficientnet_b0/comparison.csv        (bảng so sánh 2 model)
  figures/efficientnet_b0/confusion_matrix.png
  figures/efficientnet_b0/learning_curves.png   (so với CustomCNN nếu có history)

Chạy từ thư mục gốc project (sau khi train xong):
    .venv312\\Scripts\\python src\\evaluate_efficientnet.py
"""

import json
import time

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data import create_dataloaders
from evaluate import evaluate_test, print_metrics
from model import CustomCNN
from efficientnet_common import (
    NUM_CLASSES,
    CLASS_NAMES,
    OUTPUT_DIR,
    FIGURE_DIR,
    BEST_MODEL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    BASELINE_METRICS_PATH,
    BASELINE_HISTORY_PATH,
    build_efficientnet_b0,
    count_parameters,
    fix_manifest_paths,
)


@torch.no_grad()
def measure_fps(model, device, batch_size=16, image_size=224,
                warmup=10, iterations=50):
    """Tốc độ suy luận thuần của model (không tính thời gian đọc ảnh),
    đo giống hệt nhau cho cả hai model để so sánh công bằng."""
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return batch_size * iterations / (time.perf_counter() - start)


def compute_metrics(y_true, y_pred):
    fire_classes = [CLASS_NAMES.index("fire"), CLASS_NAMES.index("smoke_fire")]

    # Bài toán phụ: ảnh có lửa (fire hoặc smoke_fire) vs không có lửa
    true_has_fire = [int(t in fire_classes) for t in y_true]
    pred_has_fire = [int(p in fire_classes) for p in y_pred]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             zero_division=0),
        "has_fire_recall": recall_score(true_has_fire, pred_has_fire,
                                        zero_division=0),
        "has_fire_precision": precision_score(true_has_fire, pred_has_fire,
                                              zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - EfficientNet-B0",
    )
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Đã lưu confusion matrix: {path}")


def save_learning_curves():
    eff = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")
    base = None
    if BASELINE_HISTORY_PATH.exists():
        base = pd.read_csv(BASELINE_HISTORY_PATH, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        ax.plot(eff["epoch"], eff[f"train_{metric}"],
                label="EfficientNet-B0 train", color="tab:blue", ls="--")
        ax.plot(eff["epoch"], eff[f"val_{metric}"],
                label="EfficientNet-B0 val", color="tab:blue")

        if base is not None:
            ax.plot(base["epoch"], base[f"train_{metric}"],
                    label="CustomCNN train", color="tab:orange", ls="--")
            ax.plot(base["epoch"], base[f"val_{metric}"],
                    label="CustomCNN val", color="tab:orange")

        finetune_rows = eff[eff["phase"] == "finetune"]
        if len(finetune_rows):
            ax.axvline(finetune_rows["epoch"].iloc[0] - 0.5,
                       color="gray", ls=":", label="Bắt đầu fine-tune")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = FIGURE_DIR / "learning_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Đã lưu learning curves: {path}")


def build_comparison(eff_metrics, device):
    eff_history = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")

    eff_model = build_efficientnet_b0(pretrained=False).to(device)
    cnn_model = CustomCNN(num_classes=NUM_CLASSES).to(device)

    rows = {
        "EfficientNet-B0": {
            **eff_metrics,
            "params_M": count_parameters(eff_model) / 1e6,
            "epochs_trained": len(eff_history),
            "train_time_min": eff_history["elapsed_seconds"].sum() / 60,
            "fps": measure_fps(eff_model, device),
        }
    }

    cnn_row = {
        "params_M": count_parameters(cnn_model) / 1e6,
        "fps": measure_fps(cnn_model, device),
    }
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, encoding="utf-8") as file:
            cnn_row.update(json.load(file))
    if BASELINE_HISTORY_PATH.exists():
        base_history = pd.read_csv(BASELINE_HISTORY_PATH, encoding="utf-8-sig")
        cnn_row["epochs_trained"] = len(base_history)
        cnn_row["train_time_min"] = base_history["elapsed_seconds"].sum() / 60
    rows["CustomCNN"] = cnn_row

    table = pd.DataFrame(rows).T
    columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
               "has_fire_precision", "has_fire_recall",
               "params_M", "epochs_trained", "train_time_min", "fps"]
    table = table[[c for c in columns if c in table.columns]]
    table = table.apply(pd.to_numeric).round(4)

    table.to_csv(COMPARISON_PATH, encoding="utf-8-sig")

    print("\nBẢNG SO SÁNH")
    print("-" * 50)
    print(table.T.to_string())
    print(f"\nĐã lưu bảng so sánh: {COMPARISON_PATH}")
    print("(has_fire_* của CustomCNN để trống vì evaluate.py của nhóm không "
          "tính; FPS hai model đo cùng điều kiện trên máy này.)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders()
    fix_manifest_paths(test_loader)

    model = build_efficientnet_b0(pretrained=False).to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded best checkpoint từ epoch {checkpoint['epoch']} "
          f"({checkpoint['phase']})")
    print(f"Best val loss    : {checkpoint['val_loss']:.4f}")
    print(f"Best val accuracy: {checkpoint['val_accuracy']:.4f}")

    y_true, y_pred = evaluate_test(model, test_loader, device)

    print_metrics(y_true, y_pred)       # in giống hệt CustomCNN

    metrics = compute_metrics(y_true, y_pred)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(f"\nĐã lưu metrics: {METRICS_PATH}")

    save_confusion_matrix(y_true, y_pred)
    save_learning_curves()
    build_comparison(metrics, device)


if __name__ == "__main__":
    main()
