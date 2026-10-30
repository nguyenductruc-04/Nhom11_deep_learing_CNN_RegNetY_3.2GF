import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim=128,
        depth=4,
        kernel_size=7,
        patch_size=4,
        num_classes=4
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        blocks = []

        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim,
                                dim,
                                kernel_size=kernel_size,
                                groups=dim,
                                padding=kernel_size // 2
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                        )
                    ),
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=1
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )
            )

        self.blocks = nn.Sequential(
            *blocks
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            dim,
            num_classes
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.blocks(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.classifier(x)

        return x


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ConvMixer(
        dim=128,
        depth=4,
        kernel_size=7,
        patch_size=4,
        num_classes=4
    ).to(device)

    x = torch.randn(
        4,
        3,
        224,
        224
    ).to(device)

    labels = torch.tensor(
        [0, 1, 2, 3]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    outputs = model(x)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    print(
        "Device:",
        device
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        outputs.shape
    )

    print(
        "Loss:",
        loss.item()
    )

    print(
        "Trainable parameters:",
        f"{count_trainable_parameters(model):,}"
    )

    assert outputs.shape == (
        4,
        4
    )

    assert (
        model.classifier.weight.grad
        is not None
    )

    print(
        "ConvMixer smoke test thành công."
    )


if __name__ == "__main__":
    main()