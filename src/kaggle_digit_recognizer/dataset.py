from pathlib import Path
from typing import Optional, Protocol, Sized

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SizedDataset(Sized, Protocol):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class RawDigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_path: Path):
        df = pd.read_csv(data_path)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.images = (
            torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32) / 255.0
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx].view(1, 28, 28), self.labels[idx]


class DigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: SizedDataset,
        transform: Optional[transforms.Compose] = None,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
