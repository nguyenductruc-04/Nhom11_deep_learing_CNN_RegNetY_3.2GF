from pathlib import Path
from PIL import Image
import csv
from collections import Counter


# =========================================================
# ĐƯỜNG DẪN PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_IMAGES = PROJECT_ROOT / "train" / "images"
TRAIN_LABELS = PROJECT_ROOT / "train" / "labels"

TEST_IMAGES = PROJECT_ROOT / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "test" / "labels"

MANIFEST_DIR = PROJECT_ROOT / "manifests"
MANIFEST_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = MANIFEST_DIR / "data_audit.csv"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# KIỂM TRA ẢNH CÓ ĐỌC ĐƯỢC KHÔNG
# =========================================================

def check_image(image_path):

    try:
        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:
        return False


# =========================================================
# ĐỌC LABEL YOLO
# =========================================================

def read_label(label_path):

    # Không có file label
    if not label_path.exists():
        return {
            "label_exists": False,
            "label_empty": True,
            "num_boxes": 0,
            "class_ids": []
        }

    content = label_path.read_text(
        encoding="utf-8"
    ).strip()

    # File label rỗng
    if content == "":
        return {
            "label_exists": True,
            "label_empty": True,
            "num_boxes": 0,
            "class_ids": []
        }

    class_ids = []

    lines = content.splitlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            class_ids.append(class_id)

        except ValueError:
            continue

    return {
        "label_exists": True,
        "label_empty": False,
        "num_boxes": len(lines),
        "class_ids": class_ids
    }


# =========================================================
# AUDIT MỘT SPLIT
# =========================================================

def audit_split(split_name, image_dir, label_dir):

    rows = []

    class_counter = Counter()

    images = sorted([
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print()
    print("=" * 60)
    print(split_name.upper())
    print("=" * 60)

    print("Số ảnh tìm thấy:", len(images))

    readable_count = 0
    bad_count = 0
    missing_label_count = 0
    empty_label_count = 0
    total_boxes = 0


    for image_path in images:

        readable = check_image(image_path)

        if readable:
            readable_count += 1
        else:
            bad_count += 1


        label_path = (
            label_dir /
            f"{image_path.stem}.txt"
        )


        label_info = read_label(
            label_path
        )


        if not label_info["label_exists"]:
            missing_label_count += 1


        if label_info["label_empty"]:
            empty_label_count += 1


        total_boxes += (
            label_info["num_boxes"]
        )


        for class_id in label_info["class_ids"]:
            class_counter[class_id] += 1


        class_ids_string = ",".join(
            str(x)
            for x in sorted(
                set(label_info["class_ids"])
            )
        )


        rows.append({
            "split": split_name,
            "filepath": str(
                image_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "label_path": str(
                label_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "readable": readable,
            "label_exists":
                label_info["label_exists"],
            "label_empty":
                label_info["label_empty"],
            "num_boxes":
                label_info["num_boxes"],
            "class_ids":
                class_ids_string
        })


    print("Ảnh đọc được       :", readable_count)
    print("Ảnh lỗi             :", bad_count)
    print("Thiếu label         :", missing_label_count)
    print("Label rỗng          :", empty_label_count)
    print("Tổng bounding boxes :", total_boxes)

    print()
    print("Các class ID xuất hiện:")

    if len(class_counter) == 0:
        print("Không tìm thấy class nào.")
    else:

        for class_id, count in sorted(
            class_counter.items()
        ):

            print(
                f"Class {class_id}: "
                f"{count} bounding boxes"
            )

    return rows


# =========================================================
# CHẠY AUDIT
# =========================================================

all_rows = []


all_rows.extend(
    audit_split(
        "train",
        TRAIN_IMAGES,
        TRAIN_LABELS
    )
)


all_rows.extend(
    audit_split(
        "test",
        TEST_IMAGES,
        TEST_LABELS
    )
)


# =========================================================
# GHI CSV
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csvfile:

    fieldnames = [
        "split",
        "filepath",
        "label_path",
        "readable",
        "label_exists",
        "label_empty",
        "num_boxes",
        "class_ids"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


print()
print("=" * 60)
print("HOÀN THÀNH AUDIT")
print("=" * 60)

print(
    "File kết quả:",
    OUTPUT_CSV
)