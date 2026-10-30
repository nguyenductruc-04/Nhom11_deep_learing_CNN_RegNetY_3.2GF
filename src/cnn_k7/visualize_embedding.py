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
    / "embedding_heatmap.png"
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

    embeddings = {}

    def embedding_hook(
        module,
        inputs,
        output
    ):
        embeddings["late"] = (
            output
            .detach()
            .cpu()
        )

    handle = (
        model
        .embedding[1]
        .register_forward_hook(
            embedding_hook
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

    late_vectors = (
        embeddings["late"]
    )

    print(
        "Embedding shape:",
        late_vectors.shape
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    image = ax.imshow(
        late_vectors,
        aspect="auto"
    )

    ax.figure.colorbar(
        image,
        ax=ax
    )

    ax.set_xlabel(
        "Embedding dimension"
    )

    ax.set_ylabel(
        "Representative image"
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_title(
        "Late-layer Embedding Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()

    print(
        "Ảnh đại diện:"
    )

    for label, path in zip(
        labels,
        paths
    ):
        print(
            f"{label}: {path}"
        )

    print(
        "Đã lưu embedding heatmap:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()