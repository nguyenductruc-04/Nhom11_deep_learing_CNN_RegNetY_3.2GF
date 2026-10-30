from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
)

LOSS_FIGURE_PATH = (
    FIGURE_DIR
    / "learning_curve_loss.png"
)

ACCURACY_FIGURE_PATH = (
    FIGURE_DIR
    / "learning_curve_accuracy.png"
)


def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history = pd.read_csv(
        HISTORY_PATH
    )

    epochs = history["epoch"]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_FIGURE_PATH,
        dpi=200
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_FIGURE_PATH,
        dpi=200
    )

    plt.close()

    print(
        "Đã lưu loss curve:",
        LOSS_FIGURE_PATH
    )

    print(
        "Đã lưu accuracy curve:",
        ACCURACY_FIGURE_PATH
    )


if __name__ == "__main__":
    main()