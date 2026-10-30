"""
Vẽ lại Hình 7: đường cong huấn luyện của RegNetY-3.2GF (và CNN-K7 nếu có).

Cách chạy (từ thư mục gốc project RegNetY):
    .venv312\\Scripts\\python.exe src\\plot_hinh7.py ^
        --regnet outputs\\regnet_y_3_2gf\\history.csv ^
        --k7 D:\\duong_dan\\CNN_K7\\outputs\\k7\\history.csv

Bỏ --k7 nếu chỉ muốn vẽ RegNetY.
Ảnh lưu tại figures/hinh7_learning_curves.png (300 dpi).
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font có đủ dấu tiếng Việt (có sẵn khi cài matplotlib)
plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regnet", required=True, help="history.csv của RegNetY")
    parser.add_argument("--cnn", "--k7", dest="cnn",
                        help="history.csv của CNN tự xây (tùy chọn)")
    parser.add_argument("--cnn-label", default="CNN-K7",
                        help="Tên hiển thị của CNN tự xây trên chú giải")
    parser.add_argument("--out", default="figures/hinh7_learning_curves.png")
    args = parser.parse_args()

    reg = pd.read_csv(args.regnet, encoding="utf-8-sig")
    curves = [("RegNetY-3.2GF", reg, "tab:blue")]
    if args.cnn:
        cnn = pd.read_csv(args.cnn, encoding="utf-8-sig")
        curves.append((args.cnn_label, cnn, "tab:orange"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, metric, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        for name, h, color in curves:
            ax.plot(h["epoch"], h[f"train_{metric}"], ls="--", color=color,
                    label=f"{name} – train")
            ax.plot(h["epoch"], h[f"val_{metric}"], color=color,
                    label=f"{name} – validation")

        # Vạch bắt đầu fine-tune và checkpoint tốt nhất của RegNetY
        if "phase" in reg.columns and (reg["phase"] == "finetune").any():
            start = reg.loc[reg["phase"] == "finetune", "epoch"].iloc[0]
            ax.axvline(start - 0.5, color="gray", ls=":",
                       label="RegNetY bắt đầu fine-tune")
        best = reg.loc[reg["val_loss"].idxmin()]
        ax.scatter(best["epoch"], best[f"val_{metric}"], s=60, zorder=5,
                   color="tab:blue", edgecolor="black",
                   label=f"RegNetY checkpoint tốt nhất (epoch {int(best['epoch'])})")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
