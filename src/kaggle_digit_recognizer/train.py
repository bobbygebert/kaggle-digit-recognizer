import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kaggle_digit_recognizer.model import DigitRecognizer


def train(
    model: DigitRecognizer,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
