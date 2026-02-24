"""Overlap-constrained crop pair extraction.

Shared by both SimCLR and nnBYOL3D. Extracts two crops from a volume
with guaranteed minimum per-axis overlap.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

CROP_PRESETS = {
    "default": {"patch_size": (256, 256, 256), "crop_size": (96, 96, 96)},
    "large": {"patch_size": (256, 256, 256), "crop_size": (128, 128, 128)},
}


def _constrained_start(ref_start: int, max_offset: int, max_start: int) -> int:
    """Compute a random start position within max_offset of ref_start."""
    lo = max(0, ref_start - max_offset)
    hi = min(max_start, ref_start + max_offset)
    if lo > hi:
        return lo
    return int(np.random.randint(lo, hi + 1))


def extract_crop_pair(
    volume: np.ndarray,
    crop_size: int = 96,
    min_overlap_per_axis: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract two overlapping crops from a volume.

    Overlap constraint: per-axis overlap >= min_overlap_per_axis.
    For crop_size=96 and min_overlap_per_axis=0.5:
      max offset between crop1 and crop2 per axis = 48 voxels
      guaranteed shared region per axis >= 48 voxels

    Args:
        volume: [C, D, H, W] already z-scored numpy array.
        crop_size: spatial size of each crop.
        min_overlap_per_axis: minimum overlap fraction per axis (0.5 = 50%).

    Returns:
        (crop1, crop2): each [C, crop_size, crop_size, crop_size].
    """
    if volume.ndim == 3:
        volume = volume[np.newaxis]  # add channel dim

    C, D, H, W = volume.shape

    max_start_d = D - crop_size
    max_start_h = H - crop_size
    max_start_w = W - crop_size

    if max_start_d < 0 or max_start_h < 0 or max_start_w < 0:
        raise ValueError(
            f"Volume spatial dims ({D}, {H}, {W}) too small for crop_size={crop_size}"
        )

    max_offset = int(crop_size * (1.0 - min_overlap_per_axis))

    # First crop: random position
    d1 = int(np.random.randint(0, max_start_d + 1))
    h1 = int(np.random.randint(0, max_start_h + 1))
    w1 = int(np.random.randint(0, max_start_w + 1))

    # Second crop: constrained to overlap with first
    d2 = _constrained_start(d1, max_offset, max_start_d)
    h2 = _constrained_start(h1, max_offset, max_start_h)
    w2 = _constrained_start(w1, max_offset, max_start_w)

    c = crop_size
    crop1 = volume[:, d1 : d1 + c, h1 : h1 + c, w1 : w1 + c].copy()
    crop2 = volume[:, d2 : d2 + c, h2 : h2 + c, w2 : w2 + c].copy()

    return crop1, crop2
