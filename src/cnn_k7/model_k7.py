import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2)
        )


        self.block2 = self._block(
            in_channels=32,
            out_channels=64,
            pool=True
        )

        self.block3 = self._block(
            in_channels=64,
            out_channels=128,
            pool=True
        )

        self.block4 = self._block(
            in_channels=128,
            out_channels=256,
            pool=False
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.30),
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    @staticmethod
    def _block(in_channels, out_channels, pool):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]

        if pool:
            layers.append(
                nn.MaxPool2d(kernel_size=2)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)
        x = x.flatten(1)

        x = self.embedding(x)

        logits = self.classifier(x)

        return logits


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = CustomCNN(num_classes=4).to(device)

    x = torch.randn(
        4,
        3,
        224,
        224,
        device=device
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        device=device
    )

    logits = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", logits.shape)

    assert logits.shape == (4, 4)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    print("Loss:", loss.item())

    assert model.classifier.weight.grad is not None

    total_params = count_parameters(model)

    print("Trainable parameters:", total_params)

    print("\nSmoke test model thành công.")


if __name__ == "__main__":
    main()