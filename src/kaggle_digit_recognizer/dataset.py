from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_path: Path, transform: Optional[transforms.Compose] = None):
        df = pd.read_csv(data_path)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.images = (
            torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32) / 255.0
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].view(1, 28, 28)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
