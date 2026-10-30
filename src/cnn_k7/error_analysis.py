from pathlib import Path

import torch
import matplotlib.pyplot as plt

from PIL import Image

from data import create_dataloaders
from model import CustomCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "best_model.pt"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_PATH = (
    FIGURE_DIR
    / "error_analysis_fire.png"
)


CLASS_NAMES = [
    "none",
    "smoke",
    "fire",
    "smoke_fire",
]

NUM_CLASSES = 4
MAX_IMAGES = 12


@torch.no_grad()
def collect_fire_errors(
    model,
    test_loader,
    device
):
    errors = []

    model.eval()

    for images, labels, paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):
            true_label = int(
                labels[i].item()
            )

            predicted_label = int(
                predictions[i].item()
            )

            if (
                true_label == 2
                and predicted_label != true_label
            ):
                errors.append(
                    (
                        paths[i],
                        true_label,
                        predicted_label
                    )
                )

            if len(errors) >= MAX_IMAGES:
                return errors

    return errors


def plot_errors(
    errors
):
    if len(errors) == 0:
        print(
            "Không tìm thấy ảnh fire bị dự đoán sai."
        )
        return

    rows = 3
    cols = 4

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 9)
    )

    axes = axes.flatten()

    for index, ax in enumerate(axes):
        if index >= len(errors):
            ax.axis("off")
            continue

        path, true_label, predicted_label = (
            errors[index]
        )

        image = Image.open(
            path
        ).convert("RGB")

        ax.imshow(
            image
        )

        ax.set_title(
            "True: "
            f"{CLASS_NAMES[true_label]}\n"
            "Pred: "
            f"{CLASS_NAMES[predicted_label]}"
        )

        ax.axis("off")

    fig.suptitle(
        "Misclassified Fire Images"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()


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

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
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
        "Loaded checkpoint epoch:",
        checkpoint["epoch"]
    )

    errors = collect_fire_errors(
        model,
        test_loader,
        device
    )

    print(
        "Số lỗi fire thu thập được:",
        len(errors)
    )

    prediction_counts = {}

    for _, _, predicted_label in errors:
        predicted_name = (
            CLASS_NAMES[
                predicted_label
            ]
        )

        prediction_counts[
            predicted_name
        ] = (
            prediction_counts.get(
                predicted_name,
                0
            )
            + 1
        )

    print(
        "Các lớp fire bị nhầm sang:"
    )

    for class_name, count in (
        prediction_counts.items()
    ):
        print(
            f"{class_name}: {count}"
        )

    plot_errors(
        errors
    )

    if len(errors) > 0:
        print(
            "Đã lưu hình phân tích lỗi:",
            OUTPUT_PATH
        )


if __name__ == "__main__":
    main()