from pathlib import Path

import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "DigitRecognizer":
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
