"""CF2-WRD manuscript-aligned reference implementation."""

from .model import CF2WRD, build_model
from .data import WheatRustDataset, PairedImageTransform
from .preprocessing import LabKMeansLesionPreprocessor
from .selector import fit_local_complementary_selector

__all__ = [
    "CF2WRD",
    "build_model",
    "WheatRustDataset",
    "PairedImageTransform",
    "LabKMeansLesionPreprocessor",
    "fit_local_complementary_selector",
]
