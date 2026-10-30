from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_DIR = PROJECT_ROOT / "manifests"

TRAIN_CSV = MANIFEST_DIR / "train.csv"
VAL_CSV = MANIFEST_DIR / "val.csv"
TEST_CSV = MANIFEST_DIR / "test.csv"


IMAGE_SIZE = 224
BATCH_SIZE = 16


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


class DFireDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["filepath"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path


def create_dataloaders():
    train_dataset = DFireDataset(
        TRAIN_CSV,
        transform=train_transform,
    )

    val_dataset = DFireDataset(
        VAL_CSV,
        transform=eval_transform,
    )

    test_dataset = DFireDataset(
        TEST_CSV,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels, paths = next(iter(train_loader))

    print("Train batch images:", images.shape)
    print("Train batch labels:", labels.shape)

    print("Label min:", labels.min().item())
    print("Label max:", labels.max().item())

    print("Ảnh đầu tiên:", paths[0])

    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)

    assert labels.ndim == 1
    assert labels.min().item() >= 0
    assert labels.max().item() <= 3

    print("\nDataLoader kiểm tra thành công.")


if __name__ == "__main__":
    main()