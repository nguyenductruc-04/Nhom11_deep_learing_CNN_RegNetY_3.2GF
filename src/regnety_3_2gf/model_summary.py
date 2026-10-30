import torch

from model import CustomCNN


NUM_CLASSES = 4
IMAGE_SIZE = 224


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

    model = CustomCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    model.eval()

    print("Device:", device)

    print("\n=== MODEL ARCHITECTURE ===")
    print(model)

    print("\n=== TRAINABLE PARAMETERS ===")
    total_parameters = count_trainable_parameters(
        model
    )

    print(
        "Trainable parameters:",
        f"{total_parameters:,}"
    )

    x = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(device)

    print("\n=== TENSOR SHAPES ===")
    print(
        "Input:",
        x.shape
    )

    with torch.no_grad():
        x = model.block1(x)
        print(
            "After block1:",
            x.shape
        )

        x = model.block2(x)
        print(
            "After block2:",
            x.shape
        )

        x = model.block3(x)
        print(
            "After block3:",
            x.shape
        )

        x = model.block4(x)
        print(
            "After block4:",
            x.shape
        )

        x = model.gap(x)
        print(
            "After GAP:",
            x.shape
        )

        x = torch.flatten(
            x,
            1
        )
        print(
            "After flatten:",
            x.shape
        )

        x = model.embedding(x)
        print(
            "After embedding:",
            x.shape
        )

        x = model.classifier(x)
        print(
            "Output logits:",
            x.shape
        )


if __name__ == "__main__":
    main()