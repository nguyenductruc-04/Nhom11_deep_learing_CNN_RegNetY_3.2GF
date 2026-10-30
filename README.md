# Phát hiện cháy từ ảnh bằng CNN – Nhóm 12

Đề tài môn Học sâu. Phân loại ảnh thành 4 lớp **none / smoke / fire / smoke_fire** trên bộ dữ liệu D-Fire, so sánh CNN tự xây dựng với mô hình học chuyển giao RegNetY-3.2GF.

- **GVHD:** Trần Thị Minh Hạnh
- **Thành viên:** Nguyễn Đức Trúc, Nguyễn Tấn Vỹ, Nguyễn Trường Thọ

## 1. Báo cáo và slide

| Tài liệu | Đường dẫn |
|---|---|
| Slide thuyết trình | [`docs/slide.pdf`](docs/slide.pdf) |
| Báo cáo | [`docs/bao_cao.pdf`](docs/bao_cao.pdf) |

## 2. Dữ liệu

- **Bộ dữ liệu gốc:** D-Fire – https://github.com/gaiasd/DFireDataset
- **Dữ liệu đã chia của nhóm (Google Drive):** `https://drive.google.com/file/d/1tJmSiZk5HkXaDKclC4Wc6NFoRH-U821J/view`
- Tổng 21.527 ảnh: Train 13.779 / Val 3.442 / Test 4.306
- Các file chia tập (`train.csv`, `val.csv`, `test.csv`) và `class_to_idx.json` nằm trong thư mục `data/`

> Ảnh không được đưa lên GitHub vì dung lượng lớn. Tải từ link trên rồi đặt vào thư mục `data/`.

## 3. Kết quả trên tập test (4.306 ảnh)

| Mô hình | Số tham số | Accuracy | Macro F1 | Recall lớp fire |
|---|---:|---:|---:|---:|
| CNN tự xây (Conv1 3×3) | 422.788 | 82,28% | 76,08% | – |
| CNN tự xây (Conv1 7×7) | 426.628 | 82,23% | 73,75% | 34,09% |
| **RegNetY-3.2GF (transfer learning)** | 17.929.390 | **93,59%** | **90,23%** | **76,82%** |

Nhóm có lửa (fire + smoke_fire) với RegNetY-3.2GF: Recall 93,54%, Precision 96,13%.

Bảng chi tiết nằm trong `outputs/`, hình (ma trận nhầm lẫn, đường cong huấn luyện) nằm trong `figures/`.

![Đường cong huấn luyện](figures/hinh7_learning_curves.png)

## 4. Cấu trúc thư mục

```
├── configs/      # file cấu hình (.yaml)
├── data/         # file chia tập .csv (ảnh tải riêng)
├── docs/         # slide + báo cáo
├── figures/      # hình kết quả
├── outputs/      # metrics, comparison.csv, history.csv
├── src/          # mã nguồn
└── requirements.txt
```

## 5. Cách chạy

Yêu cầu Python 3.12 và GPU NVIDIA (đã thử trên RTX 4060 Laptop).

```bash
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt

# CNN tự xây
python src/train.py --config configs/baseline.yaml
python src/evaluate.py --config configs/baseline.yaml

# RegNetY-3.2GF
python src/train_transfer.py regnet_y_3_2gf
python src/evaluate_transfer.py regnet_y_3_2gf
```

## 6. Cài đặt huấn luyện RegNetY-3.2GF

- Trọng số pretrained ImageNet (IMAGENET1K_V2), thay lớp cuối bằng `Linear(1512, 4)`
- Giai đoạn 1: đóng băng backbone, học lớp cuối 3 epoch, lr = 1e-3
- Giai đoạn 2: fine-tune toàn mạng, lr = 1e-4
- AdamW (weight decay 1e-4), CrossEntropyLoss, batch 16, ảnh 224×224
- Early stopping với patience 7, lưu checkpoint theo val loss nhỏ nhất (epoch 6)
