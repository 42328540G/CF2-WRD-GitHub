from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .preprocessing import LabKMeansLesionPreprocessor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PairedImageTransform:
    """Create synchronized global/local views from one RGB image."""

    def __init__(
        self,
        input_size: int = 224,
        train: bool = False,
        lesion_preprocessor: Optional[LabKMeansLesionPreprocessor] = None,
        rotation_degrees: float = 15.0,
        horizontal_flip_p: float = 0.5,
        gaussian_blur_p: float = 0.3,
        gaussian_kernel: int = 3,
        brightness_delta: float = 0.2,
    ):
        self.input_size = input_size
        self.train = train
        self.lesion_preprocessor = lesion_preprocessor
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_p = horizontal_flip_p
        self.gaussian_blur_p = gaussian_blur_p
        self.gaussian_kernel = gaussian_kernel
        self.brightness_delta = brightness_delta

    def _augment(self, image: Image.Image) -> Image.Image:
        image = TF.resize(
            image,
            [self.input_size, self.input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if not self.train:
            return image

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        if random.random() < self.horizontal_flip_p:
            image = TF.hflip(image)
        if random.random() < self.gaussian_blur_p:
            k = int(self.gaussian_kernel)
            if k < 1:
                raise ValueError("gaussian_kernel must be >= 1")
            if k % 2 == 0:
                k += 1
            image = TF.gaussian_blur(image, kernel_size=[k, k])
        if self.brightness_delta > 0:
            factor = random.uniform(
                1.0 - self.brightness_delta, 1.0 + self.brightness_delta
            )
            image = ImageEnhance.Brightness(image).enhance(factor)
        return image

    @staticmethod
    def _tensorize(image: Image.Image) -> torch.Tensor:
        x = TF.to_tensor(image)
        return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        global_pil = self._augment(image.convert("RGB"))
        if self.lesion_preprocessor is None:
            local_pil = global_pil.copy()
        else:
            local_pil = self.lesion_preprocessor(global_pil)

        return {
            "global_image": self._tensorize(global_pil),
            "local_image": self._tensorize(local_pil),
        }


class WheatRustDataset(Dataset):
    """CSV-backed binary wheat stripe rust dataset.

    Required columns:
      image_path, label

    Optional columns:
      growth_stage, disease_stage, device, group_id
    """

    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path = ".",
        transform: Optional[PairedImageTransform] = None,
    ):
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    @staticmethod
    def _parse_label(value) -> int:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"healthy", "0", "normal"}:
                return 0
            if v in {"rust", "stripe_rust", "stripe rust", "infected", "1"}:
                return 1
        return int(value)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")

        views = (
            self.transform(image)
            if self.transform is not None
            else {"global_image": TF.to_tensor(image), "local_image": TF.to_tensor(image)}
        )
        item = {
            **views,
            "label": torch.tensor(self._parse_label(row["label"]), dtype=torch.long),
            "image_path": str(path),
        }
        for col in ("growth_stage", "disease_stage", "device", "group_id"):
            if col in self.df.columns:
                item[col] = "" if pd.isna(row[col]) else str(row[col])
        return item
