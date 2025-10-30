from torchvision import transforms


def get_training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_validation_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
