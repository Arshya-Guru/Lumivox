"""Fine-tune segmentation dataset.

Reads the first-pass FT manifest (e.g., manifests/abeta_ft_first_pass.json),
loads each patch's raw 128^3 crop and gold mask, per-sample z-score normalizes,
and applies paired spatial + image-only intensity augmentations.

Usage:
    from lumivox.data.dataset_finetune import (
        FineTuneDataset, subject_level_split, build_finetune_dataloaders,
    )

    train_loader, val_loader, val_subjects = build_finetune_dataloaders(
        manifest_path="manifests/abeta_ft_first_pass.json",
        batch_size=2,
        num_workers=8,
        val_fraction=0.2,
        seed=42,
    )
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return x - mu
    return (x - mu) / sd


def _paired_flip(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Spatial dims for [C, D, H, W] are 1, 2, 3.
    for axis in (1, 2, 3):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis])
    return image, mask


def _paired_rot90(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Rotate 0/1/2/3 times around the D axis in the (H, W) plane.
    k = int(torch.randint(0, 4, ()).item())
    if k:
        image = torch.rot90(image, k=k, dims=(2, 3))
        mask = torch.rot90(mask, k=k, dims=(2, 3))
    return image, mask


def _intensity_jitter(
    image: torch.Tensor,
    brightness: float = 0.1,
    contrast: Tuple[float, float] = (0.85, 1.15),
    gamma: Tuple[float, float] = (0.8, 1.25),
    noise_std: float = 0.03,
    p_apply: float = 0.8,
) -> torch.Tensor:
    if torch.rand(()) < p_apply:
        image = image + (torch.rand(()) * 2 - 1) * brightness
    if torch.rand(()) < p_apply:
        c = float(torch.empty(()).uniform_(contrast[0], contrast[1]))
        mu = image.mean()
        image = (image - mu) * c + mu
    if torch.rand(()) < 0.5:
        g = float(torch.empty(()).uniform_(gamma[0], gamma[1]))
        sign = image.sign()
        image = sign * image.abs().clamp_min(1e-6).pow(g)
    if torch.rand(()) < 0.2:
        image = image + torch.randn_like(image) * noise_std
    return image


class FineTuneDataset(Dataset):
    """Pairs raw crop128 + gold mask, z-scores, augments.

    Each sample is a dict:
        image:      [1, D, H, W] float32
        mask:       [1, D, H, W] float32 (binarised: > 0 -> 1)
        patch_id:   str
        subject_id: str
        dataset:    str
    """

    def __init__(
        self,
        entries: Sequence[Dict[str, Any]],
        augment: bool = True,
        repeats: int = 1,
    ):
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.entries = list(entries)
        self.augment = augment
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.entries) * self.repeats

    def _load(self, entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(entry["raw_path"]).get_fdata().astype(np.float32)
        msk = nib.load(entry["seg_gold_path"]).get_fdata().astype(np.float32)
        # Drop any singleton channel
        while img.ndim > 3 and img.shape[0] == 1:
            img = img[0]
        while msk.ndim > 3 and msk.shape[0] == 1:
            msk = msk[0]
        return img, msk

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx % len(self.entries)]
        img, msk = self._load(entry)
        img = _zscore(img)
        msk = (msk > 0).astype(np.float32)

        image = torch.from_numpy(img).unsqueeze(0)   # [1, D, H, W]
        mask = torch.from_numpy(msk).unsqueeze(0)    # [1, D, H, W]

        if self.augment:
            image, mask = _paired_flip(image, mask)
            image, mask = _paired_rot90(image, mask)
            image = _intensity_jitter(image)

        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "patch_id": entry["patch_id"],
            "subject_id": entry["subject_id"],
            "dataset": entry["dataset"],
        }


def load_manifest(manifest_path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(Path(manifest_path).read_text())
    return data.get("config", {}), data["patches"]


def subject_level_split(
    entries: Sequence[Dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 42,
    val_subjects: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Split patches into train/val by held-out subjects.

    If val_subjects is provided, it overrides val_fraction.
    """
    subjects = sorted({e["subject_id"] for e in entries})
    if val_subjects is None:
        n_val = max(1, int(round(len(subjects) * val_fraction)))
        rng = random.Random(seed)
        val_subjects = rng.sample(subjects, n_val)
    val_set = set(val_subjects)
    train = [e for e in entries if e["subject_id"] not in val_set]
    val = [e for e in entries if e["subject_id"] in val_set]
    return train, val, sorted(val_set)


def build_finetune_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 2,
    num_workers: int = 8,
    val_fraction: float = 0.2,
    val_subjects: Optional[Sequence[str]] = None,
    seed: int = 42,
    train_repeats: int = 1,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    _, entries = load_manifest(manifest_path)
    train_entries, val_entries, val_subjects_used = subject_level_split(
        entries, val_fraction=val_fraction, seed=seed, val_subjects=val_subjects,
    )

    train_ds = FineTuneDataset(train_entries, augment=augment_train, repeats=train_repeats)
    val_ds = FineTuneDataset(val_entries, augment=False, repeats=1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return train_loader, val_loader, val_subjects_used
