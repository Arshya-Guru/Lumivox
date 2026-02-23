"""Dataset loader for blosc2-compressed lightsheet patches.

Loads preprocessed patches, extracts overlap-constrained crop pairs,
and applies augmentations based on the selected method.

Data is already z-scored during preprocessing -- NO normalization here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lumivox.augmentations.crop_pair import extract_crop_pair
from lumivox.augmentations.shared_aug import SharedAugmentation, get_augmentation_config
from lumivox.augmentations.legacy_aug import LegacyAugment3D, augment_config as legacy_augment_config

try:
    import blosc2

    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker initialization for DataLoader multiprocessing."""
    if HAS_BLOSC2:
        blosc2.set_nthreads(1)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


class LumivoxDataset(Dataset):
    """Dataset for SSL pretraining on blosc2-compressed lightsheet patches.

    Returns two augmented views (crop pairs) from each patch.

    Args:
        data_dir: path to directory containing .b2nd or .npy files.
        method: 'simclr', 'nnbyol3d', or 'byol3d-legacy'.
        crop_size: spatial size of each crop.
        min_overlap_per_axis: per-axis overlap fraction for crop pairs.
    """

    def __init__(
        self,
        data_dir: str,
        method: str = "simclr",
        crop_size: int = 96,
        min_overlap_per_axis: float = 0.5,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.min_overlap = min_overlap_per_axis
        self.method = method

        data_path = Path(data_dir)
        self.file_paths: List[Path] = sorted(
            list(data_path.rglob("*.b2nd")) + list(data_path.rglob("*.npy"))
        )
        if not self.file_paths:
            raise FileNotFoundError(f"No .b2nd or .npy files found in {data_dir}")

        print(f"LumivoxDataset: found {len(self.file_paths)} files, method={method}")

        # Build augmentations based on method
        if method == "byol3d-legacy":
            self.view1_aug = LegacyAugment3D(**legacy_augment_config["view1"])
            self.view2_aug = LegacyAugment3D(**legacy_augment_config["view2"])
        else:
            aug_cfg = get_augmentation_config(method)
            self.view1_aug = SharedAugmentation(**aug_cfg["view1"])
            self.view2_aug = SharedAugmentation(**aug_cfg["view2"])

    def __len__(self) -> int:
        return len(self.file_paths)

    @staticmethod
    def _get_spatial_shape(raw_shape: tuple) -> Tuple[int, int, int]:
        shape = list(raw_shape)
        while len(shape) > 3 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3:
            raise ValueError(f"Cannot extract 3D spatial dims from shape {raw_shape}")
        return tuple(shape)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            return self._load_sample(idx)
        except Exception:
            fallback_idx = int(np.random.randint(0, len(self)))
            try:
                return self._load_sample(fallback_idx)
            except Exception:
                c = self.crop_size
                zeros = torch.zeros(1, c, c, c)
                return {"view1": zeros.clone(), "view2": zeros.clone(), "index": idx}

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[idx]

        if path.suffix == ".b2nd" and HAS_BLOSC2:
            arr = blosc2.open(str(path), mode="r", mmap_mode="r")
            vol = np.array(arr[:], dtype=np.float32, copy=True)
        elif path.suffix == ".npy":
            vol = np.load(str(path)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Squeeze leading singletons to get [D, H, W] or [C, D, H, W]
        while vol.ndim > 3 and vol.shape[0] == 1:
            vol = vol.squeeze(0)

        if vol.ndim == 3:
            vol = vol[np.newaxis]  # [1, D, H, W]

        crop1, crop2 = extract_crop_pair(
            vol, crop_size=self.crop_size, min_overlap_per_axis=self.min_overlap
        )

        crop1_t = torch.from_numpy(np.ascontiguousarray(crop1)).float()
        crop2_t = torch.from_numpy(np.ascontiguousarray(crop2)).float()

        # Apply augmentations: expects [B, C, D, H, W], we add/remove batch dim
        view1 = self.view1_aug(crop1_t.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2_t.unsqueeze(0)).squeeze(0)

        return {"view1": view1, "view2": view2, "index": idx}


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing without real data."""

    def __init__(
        self,
        num_samples: int = 100,
        volume_size: int = 32,
        crop_size: int = 16,
        method: str = "simclr",
        min_overlap_per_axis: float = 0.5,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.crop_size = crop_size
        self.min_overlap = min_overlap_per_axis

        if method == "byol3d-legacy":
            self.view1_aug = LegacyAugment3D(**legacy_augment_config["view1"])
            self.view2_aug = LegacyAugment3D(**legacy_augment_config["view2"])
        else:
            aug_cfg = get_augmentation_config(method)
            self.view1_aug = SharedAugmentation(**aug_cfg["view1"])
            self.view2_aug = SharedAugmentation(**aug_cfg["view2"])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = np.random.RandomState(idx)
        vol = gen.rand(1, self.volume_size, self.volume_size, self.volume_size).astype(
            np.float32
        )

        crop1, crop2 = extract_crop_pair(
            vol, self.crop_size, self.min_overlap
        )

        crop1_t = torch.from_numpy(crop1).float()
        crop2_t = torch.from_numpy(crop2).float()

        view1 = self.view1_aug(crop1_t.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2_t.unsqueeze(0)).squeeze(0)

        return {"view1": view1, "view2": view2, "index": idx}


def create_dataloader(
    data_dir: Optional[str] = None,
    method: str = "simclr",
    batch_size: int = 4,
    crop_size: int = 96,
    min_overlap_per_axis: float = 0.5,
    num_workers: int = 4,
    pin_memory: bool = True,
    synthetic: bool = False,
    synthetic_num_samples: int = 100,
    synthetic_volume_size: int = 32,
) -> DataLoader:
    """Create a DataLoader for SSL pretraining."""
    if synthetic or data_dir is None:
        crop = min(crop_size, synthetic_volume_size // 2)
        dataset = SyntheticDataset(
            num_samples=synthetic_num_samples,
            volume_size=synthetic_volume_size,
            crop_size=crop,
            method=method,
            min_overlap_per_axis=min_overlap_per_axis,
        )
    else:
        dataset = LumivoxDataset(
            data_dir=data_dir,
            method=method,
            crop_size=crop_size,
            min_overlap_per_axis=min_overlap_per_axis,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
