import torch
from torch.utils.data import DataLoader, Dataset

from kaggle_digit_recognizer.model import DigitRecognizer


def evaluate(
    model: DigitRecognizer,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
