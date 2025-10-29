from pathlib import Path

import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)
        return self.model(x)  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "DigitRecognizer":
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
