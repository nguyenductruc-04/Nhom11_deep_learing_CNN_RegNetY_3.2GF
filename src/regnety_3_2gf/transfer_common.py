"""
Phần dùng chung cho các model transfer learning (pretrained ImageNet).

Model hỗ trợ:
    regnet_y_3_2gf   -> RegNetY-3.2GF
    efficientnet_b0  -> EfficientNet-B0

Kết quả mỗi model lưu riêng: outputs/<tên model>/, figures/<tên model>/
-> không ghi đè kết quả của CustomCNN hay model khác.
"""

from pathlib import Path, PureWindowsPath

from torch import nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NUM_CLASSES = 4
CLASS_NAMES = ["none", "smoke", "fire", "smoke_fire"]

# Kết quả CustomCNN của nhóm
BASELINE_METRICS_PATH = PROJECT_ROOT / "outputs" / "test_metrics.json"
BASELINE_HISTORY_PATH = PROJECT_ROOT / "outputs" / "history.csv"


def _efficientnet_b0(pretrained):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def _regnet_y_3_2gf(pretrained):
    # IMAGENET1K_V2: 82.6% top-1 ImageNet
    weights = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.regnet_y_3_2gf(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)   # 1512 -> 4
    return model


# key: (tên hiển thị, hàm tạo model, tên lớp phân loại cuối)
MODELS = {
    "regnet_y_3_2gf": ("RegNetY-3.2GF", _regnet_y_3_2gf, "fc"),
    "efficientnet_b0": ("EfficientNet-B0", _efficientnet_b0, "classifier"),
}


class ModelSpec:
    def __init__(self, key):
        if key not in MODELS:
            raise SystemExit(
                f"Model '{key}' không hỗ trợ. Chọn một trong: {', '.join(MODELS)}"
            )
        self.key = key
        self.display_name, self._builder, self.head_name = MODELS[key]

        self.output_dir = PROJECT_ROOT / "outputs" / key
        self.figure_dir = PROJECT_ROOT / "figures" / key
        self.best_model_path = self.output_dir / "best_model.pt"
        self.history_path = self.output_dir / "history.csv"
        self.metrics_path = self.output_dir / "test_metrics.json"
        self.comparison_path = self.output_dir / "comparison.csv"

    def build(self, pretrained=True):
        return self._builder(pretrained)


def set_backbone_trainable(model, trainable, head_name):
    """Đóng băng / mở băng mọi lớp, trừ lớp phân loại cuối (head_name)."""
    for name, parameter in model.named_parameters():
        if not name.startswith(head_name + "."):
            parameter.requires_grad = trainable


def count_parameters(model, trainable_only=False):
    return sum(
        p.numel()
        for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


def fix_manifest_paths(*loaders):
    """
    Manifest lưu đường dẫn tuyệt đối từ máy tạo ra nó (vd: D:\\CNN\\D-Fire\\...).
    Nếu đường dẫn đó không tồn tại, ghép lại theo thư mục project:
    <PROJECT_ROOT>/<train|test>/images/<tên ảnh>. Không đổi cách chia dữ liệu.
    """
    for loader in loaders:
        dataframe = loader.dataset.dataframe

        def remap(path_str):
            if Path(path_str).exists():
                return path_str
            tail = PureWindowsPath(path_str).parts[-3:]
            return str(PROJECT_ROOT.joinpath(*tail))

        dataframe["filepath"] = dataframe["filepath"].map(remap)

        missing = [p for p in dataframe["filepath"] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Thiếu {len(missing)}/{len(dataframe)} ảnh trong manifest. "
                f"Ví dụ: {missing[0]}\n"
                "Hãy chép đủ ảnh D-Fire vào train/images và test/images."
            )
