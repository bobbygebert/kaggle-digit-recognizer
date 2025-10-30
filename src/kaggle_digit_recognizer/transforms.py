from torchvision import transforms


def get_training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_validation_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
