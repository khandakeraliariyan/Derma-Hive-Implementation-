"""PyTorch HAM10000 Dataset for our clean-room Phase 1 implementation.

This is our own implementation, not code recovered from the original repository.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from preprocessing import transform_metadata


class HAM10000Dataset(Dataset):
    """Return image, processed metadata, label, and traceability IDs."""

    def __init__(self, frame: pd.DataFrame, metadata_preprocessor, image_transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_transform = image_transform
        required = {"image_path", "image_id", "lesion_id", "label"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.metadata = transform_metadata(metadata_preprocessor, self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = Path(row["image_path"])
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read image {path}: {exc}") from exc
        if self.image_transform is not None:
            image = self.image_transform(image)
        return {
            "image": image,
            "metadata": torch.from_numpy(self.metadata[index].copy()),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": str(row["image_id"]),
            "lesion_id": str(row["lesion_id"]),
        }

