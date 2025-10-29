from torchvision import transforms


def get_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
