"""
Phần dùng chung cho mô hình EfficientNet-B0 (transfer learning).

Thiết kế để chạy song song với pipeline CustomCNN của nhóm:
- Dùng lại data.py (cùng manifest, cùng augmentation, cùng chuẩn hóa ImageNet).
- Mọi kết quả lưu vào outputs/efficientnet_b0/ và figures/efficientnet_b0/,
  KHÔNG ghi đè outputs/best_model.pt hay figures/ của CustomCNN.
"""

from pathlib import Path, PureWindowsPath

from torch import nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "efficientnet_b0"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
FIGURE_DIR = PROJECT_ROOT / "figures" / MODEL_NAME

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "comparison.csv"

# Kết quả của CustomCNN (để in bảng so sánh)
BASELINE_METRICS_PATH = PROJECT_ROOT / "outputs" / "test_metrics.json"
BASELINE_HISTORY_PATH = PROJECT_ROOT / "outputs" / "history.csv"

NUM_CLASSES = 4
CLASS_NAMES = ["none", "smoke", "fire", "smoke_fire"]


def build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    """EfficientNet-B0 pretrained ImageNet, thay lớp cuối thành num_classes."""
    weights = (
        models.EfficientNet_B0_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def set_backbone_trainable(model, trainable):
    """Đóng băng / mở băng phần backbone (model.features)."""
    for parameter in model.features.parameters():
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
    Nếu đường dẫn đó không tồn tại trên máy hiện tại, ghép lại theo thư mục
    project: <PROJECT_ROOT>/<train|test>/images/<tên ảnh>.
    Không thay đổi cách chia dữ liệu, chỉ sửa đường dẫn.
    """
    for loader in loaders:
        dataframe = loader.dataset.dataframe

        def remap(path_str):
            if Path(path_str).exists():
                return path_str
            tail = PureWindowsPath(path_str).parts[-3:]   # train, images, xxx.jpg
            return str(PROJECT_ROOT.joinpath(*tail))

        dataframe["filepath"] = dataframe["filepath"].map(remap)

        missing = [
            p for p in dataframe["filepath"]
            if not Path(p).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Thiếu {len(missing)}/{len(dataframe)} ảnh trong manifest. "
                f"Ví dụ: {missing[0]}\n"
                "Hãy chép đủ ảnh D-Fire vào thư mục train/images và "
                "test/images của project rồi chạy lại."
            )
