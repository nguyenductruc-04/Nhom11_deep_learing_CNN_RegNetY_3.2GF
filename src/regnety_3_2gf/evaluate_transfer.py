"""
Đánh giá một model transfer learning trên tập test, rồi lập bảng so sánh với
CustomCNN và mọi model transfer learning khác đã có kết quả (vd EfficientNet-B0).

Đầu ra:
  outputs/<model>/test_metrics.json
  outputs/<model>/comparison.csv
  figures/<model>/confusion_matrix.png
  figures/<model>/learning_curves.png

Chạy từ thư mục gốc project (sau khi train xong):
    .venv312\\Scripts\\python.exe src\\evaluate_transfer.py regnet_y_3_2gf
"""

import json
import sys
import time

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from data import create_dataloaders
from evaluate import evaluate_test, print_metrics
from model import CustomCNN
from transfer_common import (
    NUM_CLASSES, CLASS_NAMES, MODELS,
    BASELINE_METRICS_PATH, BASELINE_HISTORY_PATH,
    ModelSpec, count_parameters, fix_manifest_paths,
)


@torch.no_grad()
def measure_fps(model, device, batch_size=16, image_size=224,
                warmup=10, iterations=50):
    """Tốc độ suy luận thuần (không tính đọc ảnh), đo như nhau cho mọi model."""
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
    fire = [CLASS_NAMES.index("fire"), CLASS_NAMES.index("smoke_fire")]
    true_fire = [int(t in fire) for t in y_true]
    pred_fire = [int(p in fire) for p in y_pred]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "has_fire_recall": recall_score(true_fire, pred_fire, zero_division=0),
        "has_fire_precision": precision_score(true_fire, pred_fire, zero_division=0),
    }


def save_confusion_matrix(spec, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel="True label", xlabel="Predicted label",
           title=f"Confusion Matrix - {spec.display_name}")
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    plt.tight_layout()
    path = spec.figure_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Đã lưu confusion matrix: {path}")


def save_learning_curves(spec):
    """Vẽ chung CustomCNN + mọi model transfer learning đã có history."""
    curves = []
    if BASELINE_HISTORY_PATH.exists():
        curves.append(("CustomCNN",
                       pd.read_csv(BASELINE_HISTORY_PATH, encoding="utf-8-sig")))
    for key in MODELS:
        other = ModelSpec(key)
        if other.history_path.exists():
            curves.append((other.display_name,
                           pd.read_csv(other.history_path, encoding="utf-8-sig")))

    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [(axes[0], "loss", "Loss"),
                              (axes[1], "accuracy", "Accuracy")]:
        for (name, h), color in zip(curves, colors):
            ax.plot(h["epoch"], h[f"train_{metric}"], ls="--", color=color,
                    label=f"{name} train")
            ax.plot(h["epoch"], h[f"val_{metric}"], color=color,
                    label=f"{name} val")
        ax.axvline(3.5, color="gray", ls=":", label="Bắt đầu fine-tune")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = spec.figure_dir / "learning_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Đã lưu learning curves: {path}")


def build_comparison(spec, device):
    rows = {}

    # Các model transfer learning đã có kết quả
    for key in MODELS:
        other = ModelSpec(key)
        if not other.metrics_path.exists():
            continue
        with open(other.metrics_path, encoding="utf-8") as f:
            row = json.load(f)
        h = pd.read_csv(other.history_path, encoding="utf-8-sig")
        model = other.build(pretrained=False).to(device)
        row.update({
            "params_M": count_parameters(model) / 1e6,
            "epochs_trained": len(h),
            "best_epoch": int(h.loc[h["val_loss"].idxmin(), "epoch"]),
            "train_time_min": h["elapsed_seconds"].sum() / 60,
            "fps": measure_fps(model, device),
        })
        rows[other.display_name] = row
        del model

    # CustomCNN
    cnn = CustomCNN(num_classes=NUM_CLASSES).to(device)
    cnn_row = {"params_M": count_parameters(cnn) / 1e6,
               "fps": measure_fps(cnn, device)}
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, encoding="utf-8") as f:
            cnn_row.update(json.load(f))
    if BASELINE_HISTORY_PATH.exists():
        h = pd.read_csv(BASELINE_HISTORY_PATH, encoding="utf-8-sig")
        cnn_row["epochs_trained"] = len(h)
        cnn_row["best_epoch"] = int(h.loc[h["val_loss"].idxmin(), "epoch"])
        cnn_row["train_time_min"] = h["elapsed_seconds"].sum() / 60
    rows["CustomCNN"] = cnn_row

    columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
               "has_fire_precision", "has_fire_recall", "params_M",
               "epochs_trained", "best_epoch", "train_time_min", "fps"]
    table = pd.DataFrame(rows).T
    table = table[[c for c in columns if c in table.columns]]
    table = table.apply(pd.to_numeric).round(4)
    table.to_csv(spec.comparison_path, encoding="utf-8-sig")

    print("\nBẢNG SO SÁNH")
    print("-" * 60)
    print(table.T.to_string())
    print(f"\nĐã lưu bảng so sánh: {spec.comparison_path}")
    print("(FPS mọi model đo cùng điều kiện trên máy này.)")


def main(model_key):
    spec = ModelSpec(model_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.figure_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders()
    fix_manifest_paths(test_loader)

    model = spec.build(pretrained=False).to(device)
    checkpoint = torch.load(spec.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model: {spec.display_name}")
    print(f"Loaded best checkpoint từ epoch {checkpoint['epoch']} "
          f"({checkpoint['phase']})")
    print(f"Best val loss    : {checkpoint['val_loss']:.4f}")
    print(f"Best val accuracy: {checkpoint['val_accuracy']:.4f}")
    print("\nĐang dự đoán tập test (mất vài phút, không in gì trong lúc chạy)...")

    y_true, y_pred = evaluate_test(model, test_loader, device)
    print_metrics(y_true, y_pred)

    with open(spec.metrics_path, "w", encoding="utf-8") as f:
        json.dump(compute_metrics(y_true, y_pred), f, indent=4)
    print(f"\nĐã lưu metrics: {spec.metrics_path}")

    save_confusion_matrix(spec, y_true, y_pred)
    save_learning_curves(spec)
    del model
    build_comparison(spec, device)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: evaluate_transfer.py regnet_y_3_2gf | efficientnet_b0")
    main(sys.argv[1])
