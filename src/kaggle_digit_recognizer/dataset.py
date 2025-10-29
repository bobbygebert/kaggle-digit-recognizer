from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class DigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_path: Path):
        df = pd.read_csv(data_path)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.images = (
            torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32) / 255.0
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
