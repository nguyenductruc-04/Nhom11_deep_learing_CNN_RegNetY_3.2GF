from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image

from data import eval_transform
from model import CustomCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "best_model.pt"
)

TEST_CSV = (
    PROJECT_ROOT
    / "manifests"
    / "test.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_PATH = (
    FIGURE_DIR
    / "feature_maps_early.png"
)


CLASS_NAMES = [
    "none",
    "smoke",
    "fire",
    "smoke_fire",
]

NUM_CLASSES = 4


def load_representative_images():
    dataframe = pd.read_csv(
        TEST_CSV
    )

    selected_rows = []

    for class_name in CLASS_NAMES:
        class_rows = dataframe[
            dataframe["class_name"] == class_name
        ]

        row = class_rows.iloc[0]

        selected_rows.append(
            row
        )

    images = []
    labels = []
    paths = []

    for row in selected_rows:
        image_path = row["filepath"]

        image = Image.open(
            image_path
        ).convert("RGB")

        transformed = eval_transform(
            image
        )

        images.append(
            transformed
        )

        labels.append(
            row["class_name"]
        )

        paths.append(
            image_path
        )

    image_batch = torch.stack(
        images
    )

    return (
        image_batch,
        labels,
        paths
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

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
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

    model.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint["epoch"]
    )

    activations = {}

    def hook_function(
        module,
        inputs,
        output
    ):
        activations["early"] = (
            output
            .detach()
            .cpu()
        )

    handle = (
        model
        .block1[2]
        .register_forward_hook(
            hook_function
        )
    )

    image_batch, labels, paths = (
        load_representative_images()
    )

    with torch.no_grad():
        _ = model(
            image_batch.to(device)
        )

    handle.remove()

    feature_maps = (
        activations["early"]
    )

    print(
        "Feature map shape:",
        feature_maps.shape
    )

    print(
        "Số ảnh đại diện:",
        len(labels)
    )

    sample_index = 0

    maps = feature_maps[
        sample_index
    ]

    num_maps = min(
        16,
        maps.shape[0]
    )

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10, 10)
    )

    axes = axes.flatten()

    for i in range(num_maps):
        axes[i].imshow(
            maps[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Channel {i}"
        )

        axes[i].axis("off")

    for i in range(
        num_maps,
        len(axes)
    ):
        axes[i].axis("off")

    fig.suptitle(
        "Early Feature Maps - "
        f"{labels[sample_index]}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()

    print(
        "Ảnh nguồn:",
        paths[sample_index]
    )

    print(
        "Đã lưu feature maps:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()