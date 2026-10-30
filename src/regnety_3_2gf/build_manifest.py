from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGES = PROJECT_ROOT / "train" / "images"
TRAIN_LABELS = PROJECT_ROOT / "train" / "labels"

TEST_IMAGES = PROJECT_ROOT / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "test" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "manifests"

CLASS_NAMES = [
    "none",
    "smoke",
    "fire",
    "smoke_fire",
]

CLASS_TO_IDX = {
    "none": 0,
    "smoke": 1,
    "fire": 2,
    "smoke_fire": 3,
}


def yolo_to_class(label_path):
    if not label_path.exists():
        return 0, "none"

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    if len(lines) == 0:
        return 0, "none"

    class_ids = set()

    for line in lines:
        parts = line.split()

        if len(parts) > 0:
            class_ids.add(parts[0])

    if class_ids == {"0"}:
        return 1, "smoke"

    elif class_ids == {"1"}:
        return 2, "fire"

    elif class_ids == {"0", "1"}:
        return 3, "smoke_fire"

    else:
        raise ValueError(f"Nhãn không hợp lệ trong file: {label_path}")


def collect_rows(images_dir, labels_dir, split_name):
    rows = []

    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        label, class_name = yolo_to_class(label_path)

        rows.append(
            {
                "filepath": str(image_path),
                "label": label,
                "class_name": class_name,
                "split": split_name,
            }
        )

    return rows

def stratified_split(rows, val_ratio=0.20, seed=42):
    import random

    rng = random.Random(seed)

    rows_by_class = {}

    for row in rows:
        label = row["label"]

        if label not in rows_by_class:
            rows_by_class[label] = []

        rows_by_class[label].append(row)

    train_rows = []
    val_rows = []

    for label, class_rows in rows_by_class.items():
        rng.shuffle(class_rows)

        val_count = int(len(class_rows) * val_ratio)

        val_part = class_rows[:val_count]
        train_part = class_rows[val_count:]

        for row in train_part:
            row["split"] = "train"

        for row in val_part:
            row["split"] = "val"

        train_rows.extend(train_part)
        val_rows.extend(val_part)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    return train_rows, val_rows


def save_csv(rows, output_path):
    fieldnames = [
        "filepath",
        "label",
        "class_name",
        "split",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


import json
from collections import Counter


def print_distribution(name, rows):
    counts = Counter(row["class_name"] for row in rows)

    print(f"\n{name}")
    print("-" * 40)

    for class_name in CLASS_NAMES:
        print(f"{class_name:12s}: {counts.get(class_name, 0)}")

    print(f"{'TOTAL':12s}: {len(rows)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang đọc dữ liệu train...")
    full_train_rows = collect_rows(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        "train"
    )

    print("Đang chia train / validation...")
    train_rows, val_rows = stratified_split(
        full_train_rows,
        val_ratio=0.20,
        seed=42
    )

    print("Đang đọc dữ liệu test...")
    test_rows = collect_rows(
        TEST_IMAGES,
        TEST_LABELS,
        "test"
    )

    train_csv = OUTPUT_DIR / "train.csv"
    val_csv = OUTPUT_DIR / "val.csv"
    test_csv = OUTPUT_DIR / "test.csv"
    class_map_file = OUTPUT_DIR / "class_to_idx.json"

    save_csv(train_rows, train_csv)
    save_csv(val_rows, val_csv)
    save_csv(test_rows, test_csv)

    with open(class_map_file, "w", encoding="utf-8") as file:
        json.dump(
            CLASS_TO_IDX,
            file,
            ensure_ascii=False,
            indent=4
        )

    print_distribution("TRAIN", train_rows)
    print_distribution("VALIDATION", val_rows)
    print_distribution("TEST", test_rows)

    train_paths = {row["filepath"] for row in train_rows}
    val_paths = {row["filepath"] for row in val_rows}
    test_paths = {row["filepath"] for row in test_rows}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    assert set(row["label"] for row in train_rows) == {0, 1, 2, 3}
    assert set(row["label"] for row in val_rows) == {0, 1, 2, 3}
    assert set(row["label"] for row in test_rows) == {0, 1, 2, 3}

    print("\nKiểm tra trùng filepath: OK")
    print("Kiểm tra đủ 4 lớp: OK")

    print("\nCác file đã tạo:")
    print(train_csv)
    print(val_csv)
    print(test_csv)
    print(class_map_file)


if __name__ == "__main__":
    main()