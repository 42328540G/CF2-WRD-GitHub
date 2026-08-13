from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass
class LabKMeansLesionPreprocessor:
    """Generate a lesion-focused local RGB view using Lab + K-means.

    Paper-supported elements:
      * Lab color space
      * K = 3 by default
      * yellow/orange lesion candidate selection
      * 3x3 erosion + dilation

    The final manuscript specifies that lesion candidates are selected using
    Lab a*/b* color responses and region area, but it does not give every
    numerical scoring constant. This reference implementation therefore keeps
    those constants explicit and configurable rather than hiding them.
    """

    k: int = 3
    morphology_kernel: int = 3
    output: str = "masked_rgb"
    outside_scale: float = 0.15
    min_cluster_area_ratio: float = 0.002
    a_weight: float = 0.25
    attempts: int = 3
    seed: int = 42

    def _select_cluster(self, centers: np.ndarray, counts: np.ndarray) -> int:
        total = counts.sum()
        valid = counts / max(total, 1) >= self.min_cluster_area_ratio
        if not valid.any():
            valid[:] = True

        # OpenCV 8-bit Lab encodes neutral a*/b* around 128.
        # Higher b* favors yellow; a positive a* offset favors orange/red.
        yellow = centers[:, 2] - 128.0
        orange = np.maximum(centers[:, 1] - 128.0, 0.0)
        score = yellow + self.a_weight * orange
        score = np.where(valid, score, -np.inf)
        return int(np.argmax(score))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)

        cv2.setRNGSeed(self.seed)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            25,
            1.0,
        )
        _, labels, centers = cv2.kmeans(
            pixels,
            self.k,
            None,
            criteria,
            self.attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(lab.shape[:2])
        counts = np.bincount(labels.ravel(), minlength=self.k)
        lesion_id = self._select_cluster(centers, counts)

        mask = (labels == lesion_id).astype(np.uint8) * 255
        kernel = np.ones(
            (self.morphology_kernel, self.morphology_kernel), dtype=np.uint8
        )
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        if self.output == "binary_mask":
            out = np.repeat(mask[..., None], 3, axis=2)
        elif self.output == "masked_rgb":
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            scale = self.outside_scale + (1.0 - self.outside_scale) * alpha
            out = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        else:
            raise ValueError(
                f"Unsupported local preprocessing output={self.output!r}; "
                "use 'masked_rgb' or 'binary_mask'."
            )
        return Image.fromarray(out, mode="RGB")
