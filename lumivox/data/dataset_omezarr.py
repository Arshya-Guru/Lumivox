"""OME-Zarr patch dataset with lazy loading from a pre-built manifest.

Reads patches on-the-fly from OME-Zarr volumes at coordinates defined in a
manifest file (built by ``lumivox.data.manifest``).  Each ``__getitem__``
call lazily reads one region from the zarr volume, extracts an overlapping
crop pair for SSL pretraining, and applies augmentations.

The data is **not** loaded into memory up front -- zarr chunks are fetched
on demand, so even terabyte-scale datasets can be used without staging.

Usage:
    from lumivox.data.dataset_omezarr import create_omezarr_dataloader
    loader = create_omezarr_dataloader("manifests/patches.json", method="simclr")
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lumivox.augmentations.crop_pair import extract_crop_pair
from lumivox.augmentations.legacy_aug import LegacyAugment3D, augment_config as legacy_augment_config
from lumivox.augmentations.shared_aug import SharedAugmentation, get_augmentation_config
from lumivox.data.manifest import load_manifest


def _omezarr_worker_init_fn(worker_id: int) -> None:
    """Per-worker RNG seeding (zarr handles are re-opened lazily)."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


class OMEZarrPatchDataset(Dataset):
    """Lazy-loading dataset that reads patches from OME-Zarr volumes.

    Patch locations come from a manifest built by
    :func:`lumivox.data.manifest.build_patch_manifest`.  For each sample the
    dataset:

    1. Opens the subject's OME-Zarr volume (cached per worker process).
    2. Reads a ``patch_size`` voxel region centred on the manifest coordinate.
    3. Calls :func:`extract_crop_pair` to get two overlapping ``crop_size``
       sub-crops for SSL.
    4. Applies per-method augmentations.

    Args:
        manifest_path: path to manifest JSON.
        method: ``'simclr'``, ``'nnbyol3d'``, or ``'byol3d-legacy'``.
        crop_size: override the crop size stored in the manifest.
        patch_size: override the patch size stored in the manifest.
        min_overlap_per_axis: overlap fraction for crop pairs.
        normalize: ``'zscore'`` (default), ``'minmax'``, or ``None``.
    """

    def __init__(
        self,
        manifest_path: str,
        method: str = "simclr",
        crop_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        min_overlap_per_axis: float = 0.5,
        normalize: str = "zscore",
    ):
        super().__init__()
        self.manifest = load_manifest(manifest_path)
        self.patches = self.manifest["patches"]
        self.cfg = self.manifest["config"]

        self.crop_size = crop_size or self.cfg["crop_size"]
        self.patch_size = patch_size or self.cfg["patch_size"]
        self.min_overlap = min_overlap_per_axis
        self.normalize = normalize
        self.method = method

        # Per-worker volume cache -- populated lazily in __getitem__
        self._vol_cache: Dict[str, object] = {}

        # Augmentations (same as LumivoxDataset)
        if method == "byol3d-legacy":
            self.view1_aug = LegacyAugment3D(**legacy_augment_config["view1"])
            self.view2_aug = LegacyAugment3D(**legacy_augment_config["view2"])
        else:
            aug_cfg = get_augmentation_config(method)
            self.view1_aug = SharedAugmentation(**aug_cfg["view1"])
            self.view2_aug = SharedAugmentation(**aug_cfg["view2"])

        print(
            f"OMEZarrPatchDataset: {len(self.patches)} patches, "
            f"stain={self.cfg['stain']}, method={method}, "
            f"patch_size={self.patch_size}, crop_size={self.crop_size}"
        )

    # ------------------------------------------------------------------
    # Lazy volume access
    # ------------------------------------------------------------------

    def _get_volume(self, zarr_path: str, stain_channel: int, zarr_source: str = "resampled"):
        """Open (or retrieve from cache) a ZarrNii volume.

        For full-res zarr files, ``downsample_near_isotropic=True`` is used
        so that zarrnii automatically picks the pyramid level closest to
        isotropic voxels.
        """
        cache_key = f"{zarr_path}::ch{stain_channel}"
        if cache_key not in self._vol_cache:
            from zarrnii import ZarrNii

            kwargs = dict(channels=[stain_channel])
            if zarr_source == "fullres":
                kwargs["downsample_near_isotropic"] = True

            znimg = ZarrNii.from_ome_zarr(zarr_path, **kwargs)
            self._vol_cache[cache_key] = znimg
        return self._vol_cache[cache_key]

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _crop_by_voxel(self, znimg, center_vox: list) -> np.ndarray:
        """Extract a patch by direct voxel indexing (bypasses crop_centered).

        Used for resampled zarr files where the affine metadata is unreliable
        for crop_centered.  ``center_vox`` is in (Z, Y, X) order matching the
        zarr spatial axes.
        """
        darr = znimg.darr  # (C, Z, Y, X) or (1, Z, Y, X)
        half = self.patch_size // 2

        cz = int(round(center_vox[0]))
        cy = int(round(center_vox[1]))
        cx = int(round(center_vox[2]))

        z0 = max(0, cz - half)
        z1 = min(darr.shape[-3], cz + half)
        y0 = max(0, cy - half)
        y1 = min(darr.shape[-2], cy + half)
        x0 = max(0, cx - half)
        x1 = min(darr.shape[-1], cx + half)

        return darr[0, z0:z1, y0:y1, x0:x1].compute().astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            return self._load_sample(idx)
        except Exception:
            # Fall back to a random other patch on error
            fallback = int(np.random.randint(0, len(self)))
            try:
                return self._load_sample(fallback)
            except Exception:
                c = self.crop_size
                zeros = torch.zeros(1, c, c, c)
                return {"view1": zeros.clone(), "view2": zeros.clone(), "index": idx}

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.patches[idx]
        zarr_path = entry["zarr_path"]
        zarr_source = entry.get("zarr_source", "resampled")
        stain_channel = entry.get("stain_channel", 0)

        znimg = self._get_volume(zarr_path, stain_channel, zarr_source)

        if "center_vox" in entry:
            # Direct voxel slicing (resampled zarr — no crop_centered)
            vol = self._crop_by_voxel(znimg, entry["center_vox"])
        else:
            # Physical coordinate cropping (full-res zarr)
            center = tuple(entry["center_phys"])
            ps = self.patch_size
            patch_zn = znimg.crop_centered(center, patch_size=(ps, ps, ps))
            vol = patch_zn.darr.compute().astype(np.float32)

        # Squeeze to [C, D, H, W]
        while vol.ndim > 4 and vol.shape[0] == 1:
            vol = vol.squeeze(0)
        if vol.ndim == 3:
            vol = vol[np.newaxis]

        # Normalise
        if self.normalize == "zscore":
            std = vol.std()
            if std > 1e-8:
                vol = (vol - vol.mean()) / std
        elif self.normalize == "minmax":
            vmin, vmax = vol.min(), vol.max()
            if vmax - vmin > 1e-8:
                vol = (vol - vmin) / (vmax - vmin)

        # Pad if patch came back smaller than crop_size (near volume edges)
        spatial = vol.shape[1:]
        if any(s < self.crop_size for s in spatial):
            c = self.crop_size
            padded = np.zeros((vol.shape[0], c, c, c), dtype=np.float32)
            slices = tuple(slice(0, min(s, c)) for s in spatial)
            padded[:, slices[0], slices[1], slices[2]] = (
                vol[:, slices[0], slices[1], slices[2]]
            )
            vol = padded

        # Overlapping crop pair for SSL
        crop1, crop2 = extract_crop_pair(
            vol, crop_size=self.crop_size, min_overlap_per_axis=self.min_overlap,
        )

        crop1_t = torch.from_numpy(np.ascontiguousarray(crop1)).float()
        crop2_t = torch.from_numpy(np.ascontiguousarray(crop2)).float()

        view1 = self.view1_aug(crop1_t.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2_t.unsqueeze(0)).squeeze(0)

        return {"view1": view1, "view2": view2, "index": idx}


# ---------------------------------------------------------------------------
# Convenience DataLoader factory
# ---------------------------------------------------------------------------

def create_omezarr_dataloader(
    manifest_path: str,
    method: str = "simclr",
    batch_size: int = 4,
    crop_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    min_overlap_per_axis: float = 0.5,
    normalize: str = "zscore",
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for SSL pretraining from an OME-Zarr manifest."""
    dataset = OMEZarrPatchDataset(
        manifest_path=manifest_path,
        method=method,
        crop_size=crop_size,
        patch_size=patch_size,
        min_overlap_per_axis=min_overlap_per_axis,
        normalize=normalize,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_omezarr_worker_init_fn,
    )
