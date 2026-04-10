"""Build WT-normalized fine-tuning ground truth for Abeta plaque segmentation.

For each disease-animal patch in ft/Abeta/manifest.json (excluding patches that
came from WT animals), this script:

  1. Locates the matching WT reference brain(s) for the patch's batch.
  2. Computes (and caches) an N4 bias-field-corrected version of each WT volume
     using SimpleITK on a low-resolution OME-Zarr pyramid level. The bias field
     is small and lightweight to store; it is sampled trilinearly when patches
     are pulled.
  3. Extracts a 256³ window from the disease zarr at the patch center, and a
     matching 256³ window from each WT zarr at the same physical coordinate.
  4. Computes within-mask mu/sigma from the bias-corrected WT window(s),
     pools them when multiple WT animals are available, and z-score normalises
     the disease window.
  5. Crops the central 128³ from BOTH the raw disease window and the normalized
     window. Saves both to disk side-by-side with intuitive naming.
  6. Runs blur -> multi-Otsu (k=2 and k=3) -> connected components -> size
     filter on the *normalized* central 128³ to produce two candidate binary
     masks for human review.
  7. Writes a per-patch QC figure showing raw, normalized, both Otsu masks and
     the connected components.

Outputs: ft_normalized/Abeta/{dataset}/{subject}/{subject}_patchNN_*.{nii.gz,png}
plus a wt_references/ subdirectory holding cached N4 bias fields and a
manifest.json describing every patch and which WT(s) it was normalized against.

Usage:
    pixi run python scripts/build_finetune_normalized.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Lumivox imports (for the manifest helpers + atlas reference)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dask.array as da
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import threshold_multiotsu

import cc3d  # connected-components-3d
from zarrnii import ZarrNii, ZarrNiiAtlas

from lumivox.data.manifest import resolve_stain_channel, PREFERRED_SPIMQUANT


# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXISTING_MANIFEST = REPO_ROOT / "ft" / "Abeta" / "manifest.json"
OUTPUT_ROOT = REPO_ROOT / "ft_normalized" / "Abeta"
WT_CACHE_DIR = OUTPUT_ROOT / "wt_references"

PATCH_WINDOW = 256   # full window pulled from zarr (used to compute mu/sigma)
CROP_SIZE = 128      # final central crop saved to disk

# Segmentation hyperparameters
GAUSSIAN_SIGMA = 0.7    # voxels — light smoothing on the normalized z-score field
MIN_OBJECT_VOXELS = 15  # ~6-8um equivalent at 4um voxels; below = noise / single voxels
CC_CONNECTIVITY = 26    # 3D 26-connectivity

# N4 cache parameters.
# The resampled OME-Zarr usually stores only levels 0, 1, 2 on disk; ZarrNii
# will *synthesize* higher levels from level 0 if you ask for them, which is
# extremely slow (whole-volume downsample on the fly). So we always use the
# highest disk-resident level we can find and then further block-pool with
# dask.coarsen to keep the N4 input small. The bias field is smooth so this
# is plenty of resolution.
N4_BASE_LEVEL = 2          # highest pyramid level we ever read from zarr
N4_EXTRA_COARSEN = 4       # additional block-mean shrink applied after the read
N4_ITERATIONS = [30, 20]   # 2 multires levels; bias is smooth so this is enough
N4_CONVERGENCE_THRESHOLD = 1e-4

# WT subjects per batch (the dataset name as it appears in the existing manifest)
WT_SUBJECTS: Dict[str, List[str]] = {
    "mouse_app_lecanemab_batch2": ["sub-AS118M9"],                                # ApoE3 NegCtrl
    "mouse_app_lecanemab_batch3": ["sub-AS161F3", "sub-AS164F5", "sub-AS168F1"],  # ApoE3+PBS
    "mouse_app_lecanemab_ki3_batch1": ["sub-AS7F3"],                              # WT N control
    "mouse_app_lecanemab_ki3_batch2": ["sub-AS7F1"],                              # WT N control
    "mouse_app_lecanemab_ki3_batch3": ["sub-C57BL6"],                             # C57BL6 N control
    "mouse_app_vaccine_batch": [],                                                # no in-batch WT
}

# The vaccine batch in the existing manifest is stored as
# 'mouse_app_vaccine_batch' but the actual directory on disk is
# 'mouse_app_vaccine_batch1'. The other datasets are 1:1.
DATASET_DIR_OVERRIDES: Dict[str, str] = {
    "mouse_app_vaccine_batch": "mouse_app_vaccine_batch1",
}

DATASET_ROOT = "/nfs/trident3/lightsheet/prado"


# ---------------------------------------------------------------------------
# Path / WT discovery helpers
# ---------------------------------------------------------------------------

def dataset_path(dataset_name: str) -> Path:
    """Map manifest dataset name -> on-disk dataset root."""
    real = DATASET_DIR_OVERRIDES.get(dataset_name, dataset_name)
    return Path(DATASET_ROOT) / real


def is_wt_subject(dataset_name: str, subject_id: str) -> bool:
    return subject_id in WT_SUBJECTS.get(dataset_name, [])


def _max_disk_level(zarr_path: str, cap: int = 99) -> int:
    """Return the highest pyramid level stored on disk (capped at ``cap``).

    The OME-Zarr generated by the resampling pipeline only stores a few
    pyramid levels (typically 0..2). ZarrNii will *synthesize* higher
    levels by downsampling level 0 lazily, which is unbearably slow on
    multi-billion-voxel volumes — so we always read from the highest
    on-disk level instead.
    """
    import zarr as _zarr
    store = _zarr.open(zarr_path, mode="r")
    available = []
    for k in store:
        if k.isdigit():
            available.append(int(k))
    if not available:
        return 0
    return min(max(available), cap)


def _find_zarr(micr_dir: Path, suffix: str) -> Optional[Path]:
    """Find a zarr in a micr/ directory matching a suffix, preferring standard acq."""
    candidates = sorted(micr_dir.glob(f"*{suffix}"))
    standard = [c for c in candidates if "45deg" not in c.name and "90deg" not in c.name]
    chosen = standard or candidates
    return chosen[0] if chosen else None


def find_subject_paths(dataset_name: str, subject_id: str) -> Dict[str, Optional[str]]:
    """Locate the resampled zarr, full-res zarr, and dseg for a subject."""
    ds_root = dataset_path(dataset_name)
    bids = ds_root / "bids"

    fullres_micr = bids / subject_id / "micr"
    fullres_zarr = _find_zarr(fullres_micr, "_SPIM.ome.zarr") if fullres_micr.exists() else None

    resampled_micr = bids / "derivatives" / "resampled" / subject_id / "micr"
    res_zarr = _find_zarr(resampled_micr, "_res-4um_SPIM.ome.zarr") if resampled_micr.exists() else None

    sidecar = None
    if res_zarr is not None:
        sc = Path(str(res_zarr) + ".json")
        sidecar = sc if sc.exists() else None

    # Locate the dseg (used as brain mask). Reuse PREFERRED_SPIMQUANT keys when
    # available; fall back to globbing.
    dseg_path = None
    spimquant_dir = None
    for preferred in PREFERRED_SPIMQUANT.get(dataset_name, []):
        candidate = ds_root / "derivatives" / preferred
        if candidate.is_dir():
            spimquant_dir = candidate
            break
    if spimquant_dir is None:
        for c in sorted(ds_root.glob("derivatives/spimquant*")):
            if c.is_dir():
                spimquant_dir = c
                break
    if spimquant_dir is not None:
        sm = spimquant_dir / subject_id / "micr"
        if sm.exists():
            ds_candidates = sorted(sm.glob("*_seg-roi22_from-ABAv3_*_desc-deform_dseg.nii.gz"))
            if ds_candidates:
                dseg_path = ds_candidates[0]

    return {
        "fullres_zarr": str(fullres_zarr) if fullres_zarr else None,
        "resampled_zarr": str(res_zarr) if res_zarr else None,
        "sidecar": str(sidecar) if sidecar else None,
        "dseg_path": str(dseg_path) if dseg_path else None,
    }


# ---------------------------------------------------------------------------
# WT reference (N4 bias field cached, sampled trilinearly at full-res patches)
# ---------------------------------------------------------------------------

@dataclass
class WTReference:
    """Holds enough state to extract bias-corrected 256³ windows from a WT brain.

    The bias field lives in a small downsampled grid (level ``N4_BASE_LEVEL``
    further reduced by ``N4_EXTRA_COARSEN``). It is sampled trilinearly with
    map_coordinates when a patch is requested.
    """
    dataset_name: str
    subject_id: str
    fullres_zarr: str
    resampled_zarr: str
    sidecar: str
    dseg_path: Optional[str]
    stain_channel: int
    znimg_full: ZarrNii            # level-0 (4um) lazy dask
    fullres_inv: np.ndarray        # 4x4 affine inverse mapping phys -> fullres voxel
    res_scale: np.ndarray          # (3,) fullres voxel -> resampled voxel
    bias_field: np.ndarray         # float32 (Z, Y, X) bias field
    n4_level_scale_zyx: np.ndarray  # (3,) factor mapping level-0 voxels -> bias voxels
    global_mu: float                # whole-brain mean of bias-corrected intensity
    global_sigma: float             # whole-brain std of bias-corrected intensity


def _coarsen_to_array(darr: "da.Array", factor: int) -> np.ndarray:
    """Crop to a multiple of ``factor`` and block-mean coarsen by that factor.

    ``darr`` is a 3D dask array (Z, Y, X). Returns an in-memory float32 numpy
    array of shape (Z//factor, Y//factor, X//factor).
    """
    Z, Y, X = darr.shape
    Z2 = (Z // factor) * factor
    Y2 = (Y // factor) * factor
    X2 = (X // factor) * factor
    cropped = darr[:Z2, :Y2, :X2]
    coarse = da.coarsen(np.mean, cropped, {0: factor, 1: factor, 2: factor})
    return coarse.compute().astype(np.float32)


def _compute_bias_field_at_level(
    arr: np.ndarray,
    iterations: List[int] = N4_ITERATIONS,
    convergence_threshold: float = N4_CONVERGENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run SimpleITK N4 on a small in-memory volume.

    ``arr`` should already be coarsened to a tractable size (a few million
    voxels). Returns (bias_field, corrected, global_mu, global_sigma).
    Global mu/sigma are computed over the within-mask voxels of the
    bias-corrected image and are used as a fallback when patch-matched
    window stats are unavailable.
    """

    # SimpleITK has a hard requirement that mask voxels are >0 inside the
    # brain. We can let it use Otsu by passing only the image, but the helper
    # complains in newer SITK versions. So generate an Otsu-style mask
    # ourselves: anything above the lower 5% of nonzero voxels.
    nonzero = arr[arr > 0]
    if nonzero.size > 1000:
        thresh = float(np.percentile(nonzero, 5))
    else:
        thresh = 0.0
    mask = (arr > thresh).astype(np.uint8)

    # Build SITK images. We treat voxel space as identity (not in mm) — N4
    # only cares about field smoothness which is voxel-relative.
    img_sitk = sitk.GetImageFromArray(arr)
    mask_sitk = sitk.GetImageFromArray(mask)

    # SITK N4 wants its image to be float
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(iterations)
    n4.SetConvergenceThreshold(convergence_threshold)

    corrected_sitk = n4.Execute(img_sitk, mask_sitk)
    corrected = sitk.GetArrayFromImage(corrected_sitk).astype(np.float32)

    # Bias field = original / corrected (where corrected is nonzero).
    # Outside the mask we set bias=1 so division is a no-op.
    bias = np.ones_like(arr, dtype=np.float32)
    valid = (mask > 0) & (corrected > 0)
    bias[valid] = arr[valid] / corrected[valid]

    # Global within-brain mu/sigma over the bias-corrected intensity.
    if valid.sum() > 1000:
        vals = corrected[valid].astype(np.float32)
        global_mu = float(vals.mean())
        global_sigma = float(vals.std())
    else:
        global_mu = 0.0
        global_sigma = 1.0
    return bias, corrected, global_mu, global_sigma


def load_wt_reference(
    dataset_name: str,
    subject_id: str,
    stain: str = "Abeta",
    cache_dir: Path = WT_CACHE_DIR,
) -> WTReference:
    """Resolve, optionally compute+cache N4 bias field, and return a WTReference."""
    paths = find_subject_paths(dataset_name, subject_id)
    if not paths["resampled_zarr"] or not paths["sidecar"] or not paths["fullres_zarr"]:
        raise FileNotFoundError(
            f"WT subject {dataset_name}/{subject_id} missing zarr/sidecar: {paths}"
        )

    stain_channel = resolve_stain_channel(
        paths["resampled_zarr"], stain, paths["fullres_zarr"]
    )

    # Coordinate chain (same as manifest.py)
    znimg_full_res = ZarrNii.from_ome_zarr(paths["resampled_zarr"], channels=[stain_channel])
    znimg_fullres = ZarrNii.from_ome_zarr(paths["fullres_zarr"], channels=[0])
    fullres_inv = np.array(znimg_fullres.affine.invert().matrix, dtype=np.float64)
    with open(paths["sidecar"]) as f:
        sc = json.load(f)["fullres_to_resampled"]["scale"]
    res_scale = np.array([sc["z"], sc["y"], sc["x"]], dtype=np.float64)

    # Pick the highest pyramid level that actually exists on disk so the
    # downsample isn't synthesized from level 0.
    base_level = _max_disk_level(paths["resampled_zarr"], cap=N4_BASE_LEVEL)
    znimg_low = ZarrNii.from_ome_zarr(
        paths["resampled_zarr"], channels=[stain_channel], level=base_level
    )
    full_shape = znimg_full_res.darr.shape[1:]   # drop channel
    base_shape = znimg_low.darr.shape[1:]

    # Cache the bias field as a NIfTI for napari inspection (dummy 4um affine)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = f"{dataset_name}__{subject_id}__{stain}__lvl{base_level}x{N4_EXTRA_COARSEN}"
    bias_path = cache_dir / f"{cache_stem}_n4_biasfield.nii.gz"
    corrected_lowres_path = cache_dir / f"{cache_stem}_n4_corrected_lowres.nii.gz"
    meta_path = cache_dir / f"{cache_stem}_n4_meta.json"

    if bias_path.exists() and meta_path.exists():
        bias_field = nib.load(str(bias_path)).get_fdata().astype(np.float32)
        with open(meta_path) as f:
            meta = json.load(f)
        # Sanity-check shape consistency in case the level changed
        if list(bias_field.shape) != list(meta.get("bias_shape_zyx", [])):
            print(f"  WARN: cached bias shape mismatch for {subject_id}, recomputing")
            bias_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        elif "global_mu" not in meta or "global_sigma" not in meta:
            # Older cache without global stats — recompute by reloading the
            # corrected lowres NIfTI sitting next to the bias field.
            print(f"  [cache++] {dataset_name}/{subject_id} adding global mu/sigma to old cache")
            corrected_lowres_path = cache_dir / f"{cache_stem}_n4_corrected_lowres.nii.gz"
            corrected = nib.load(str(corrected_lowres_path)).get_fdata().astype(np.float32)
            mask = corrected > 0
            if mask.sum() > 1000:
                vals = corrected[mask]
                meta["global_mu"] = float(vals.mean())
                meta["global_sigma"] = float(vals.std())
            else:
                meta["global_mu"] = 0.0
                meta["global_sigma"] = 1.0
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            n4_scale_zyx = np.array(meta["n4_level_scale_zyx"], dtype=np.float64)
            return WTReference(
                dataset_name=dataset_name,
                subject_id=subject_id,
                fullres_zarr=paths["fullres_zarr"],
                resampled_zarr=paths["resampled_zarr"],
                sidecar=paths["sidecar"],
                dseg_path=paths["dseg_path"],
                stain_channel=stain_channel,
                znimg_full=znimg_full_res,
                fullres_inv=fullres_inv,
                res_scale=res_scale,
                bias_field=bias_field,
                n4_level_scale_zyx=n4_scale_zyx,
                global_mu=float(meta["global_mu"]),
                global_sigma=float(meta["global_sigma"]),
            )
        else:
            n4_scale_zyx = np.array(meta["n4_level_scale_zyx"], dtype=np.float64)
            print(
                f"  [cache] {dataset_name}/{subject_id} bias_field {bias_field.shape} "
                f"global_mu={meta['global_mu']:.1f} global_sigma={meta['global_sigma']:.1f}"
            )
            return WTReference(
                dataset_name=dataset_name,
                subject_id=subject_id,
                fullres_zarr=paths["fullres_zarr"],
                resampled_zarr=paths["resampled_zarr"],
                sidecar=paths["sidecar"],
                dseg_path=paths["dseg_path"],
                stain_channel=stain_channel,
                znimg_full=znimg_full_res,
                fullres_inv=fullres_inv,
                res_scale=res_scale,
                bias_field=bias_field,
                n4_level_scale_zyx=n4_scale_zyx,
                global_mu=float(meta["global_mu"]),
                global_sigma=float(meta["global_sigma"]),
            )

    # Compute N4 from scratch.
    # Step 1: read the base level lazily and coarsen further with dask.
    print(
        f"  [compute] {dataset_name}/{subject_id} reading level={base_level} "
        f"shape={tuple(base_shape)} -> coarsen by {N4_EXTRA_COARSEN} ..."
    )
    t0 = time.time()
    arr = _coarsen_to_array(znimg_low.darr[0], N4_EXTRA_COARSEN)
    print(
        f"    coarsen done in {time.time()-t0:.1f}s, "
        f"arr shape={arr.shape} ({arr.nbytes/1e6:.1f} MB)"
    )

    # Compute the level-0 -> bias-level scale (used by map_coordinates later).
    bias_shape = arr.shape
    n4_scale_zyx = np.array(
        [bias_shape[i] / full_shape[i] for i in range(3)], dtype=np.float64
    )

    # Step 2: run N4
    print(f"    running N4 with iters={N4_ITERATIONS} ...")
    t0 = time.time()
    bias_field, corrected_lowres, global_mu, global_sigma = _compute_bias_field_at_level(arr)
    dt = time.time() - t0
    print(
        f"    N4 done in {dt:.1f}s, bias range [{bias_field.min():.3f}, {bias_field.max():.3f}]  "
        f"global_mu={global_mu:.1f} global_sigma={global_sigma:.1f}"
    )

    # Save with dummy affine (we always use n4_level_scale_zyx + fullres_inv)
    dummy_affine = np.diag([0.004, 0.004, 0.004, 1.0])
    nib.save(nib.Nifti1Image(bias_field, dummy_affine), str(bias_path))
    nib.save(nib.Nifti1Image(corrected_lowres, dummy_affine), str(corrected_lowres_path))
    meta = {
        "dataset_name": dataset_name,
        "subject_id": subject_id,
        "stain": stain,
        "stain_channel": stain_channel,
        "n4_base_level": base_level,
        "n4_extra_coarsen": N4_EXTRA_COARSEN,
        "n4_iterations": N4_ITERATIONS,
        "bias_shape_zyx": list(bias_field.shape),
        "fullres_shape_zyx": list(full_shape),
        "n4_level_scale_zyx": [float(x) for x in n4_scale_zyx],
        "global_mu": float(global_mu),
        "global_sigma": float(global_sigma),
        "fullres_inv": fullres_inv.tolist(),
        "res_scale": res_scale.tolist(),
        "fullres_zarr": paths["fullres_zarr"],
        "resampled_zarr": paths["resampled_zarr"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return WTReference(
        dataset_name=dataset_name,
        subject_id=subject_id,
        fullres_zarr=paths["fullres_zarr"],
        resampled_zarr=paths["resampled_zarr"],
        sidecar=paths["sidecar"],
        dseg_path=paths["dseg_path"],
        stain_channel=stain_channel,
        znimg_full=znimg_full_res,
        fullres_inv=fullres_inv,
        res_scale=res_scale,
        bias_field=bias_field,
        n4_level_scale_zyx=n4_scale_zyx,
        global_mu=float(global_mu),
        global_sigma=float(global_sigma),
    )


def _phys_to_resampled_voxel(
    fullres_inv: np.ndarray, res_scale: np.ndarray, center_phys: Sequence[float]
) -> np.ndarray:
    """Map a physical-space (mm) coord to resampled-zarr voxel coords."""
    homog = np.array([center_phys[0], center_phys[1], center_phys[2], 1.0])
    vox_fullres = (fullres_inv @ homog)[:3]
    return vox_fullres * res_scale


def extract_window_from_zarr(
    znimg: ZarrNii, center_vox: np.ndarray, size: int = PATCH_WINDOW
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Extract a size³ window centered on (cz, cy, cx) from a (C, Z, Y, X) ZarrNii.

    Pads with zeros if the window extends past the volume edges. Returns the
    window AND the lo-corner offset used (so callers can also sample the bias
    field at the same coordinates).
    """
    cz, cy, cx = (int(round(v)) for v in center_vox)
    half = size // 2
    Z, Y, X = znimg.darr.shape[1:]
    z0, y0, x0 = cz - half, cy - half, cx - half
    z1, y1, x1 = z0 + size, y0 + size, x0 + size

    sz0, sz1 = max(0, z0), min(Z, z1)
    sy0, sy1 = max(0, y0), min(Y, y1)
    sx0, sx1 = max(0, x0), min(X, x1)

    window = np.zeros((size, size, size), dtype=np.float32)
    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        chunk = znimg.darr[0, sz0:sz1, sy0:sy1, sx0:sx1].compute().astype(np.float32)
        window[
            sz0 - z0 : sz0 - z0 + chunk.shape[0],
            sy0 - y0 : sy0 - y0 + chunk.shape[1],
            sx0 - x0 : sx0 - x0 + chunk.shape[2],
        ] = chunk
    return window, (z0, y0, x0)


def sample_bias_field_at_window(
    bias_field: np.ndarray,
    n4_scale_zyx: np.ndarray,
    lo_corner_full: Tuple[int, int, int],
    size: int = PATCH_WINDOW,
) -> np.ndarray:
    """Trilinearly sample a size³ region from the low-res bias field.

    The bias field is at level ``N4_LEVEL`` and ``n4_scale_zyx`` gives the
    multiplier mapping level-0 voxel index -> bias voxel index.
    """
    z0, y0, x0 = lo_corner_full
    sz, sy, sx = n4_scale_zyx

    z_idx = (z0 + np.arange(size, dtype=np.float64)) * sz
    y_idx = (y0 + np.arange(size, dtype=np.float64)) * sy
    x_idx = (x0 + np.arange(size, dtype=np.float64)) * sx

    Z, Y, X = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
    coords = np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    sampled = map_coordinates(bias_field, coords, order=1, cval=1.0, prefilter=False)
    return sampled.reshape(size, size, size).astype(np.float32)


def extract_corrected_window(
    wt: WTReference, center_phys: Sequence[float], size: int = PATCH_WINDOW
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (raw, bias_window, corrected) for a 256³ window of a WT brain
    at the given physical coordinate. ``corrected`` = raw / bias.
    """
    center_vox = _phys_to_resampled_voxel(wt.fullres_inv, wt.res_scale, center_phys)
    raw, lo = extract_window_from_zarr(wt.znimg_full, center_vox, size=size)
    bias = sample_bias_field_at_window(wt.bias_field, wt.n4_level_scale_zyx, lo, size=size)
    corrected = raw / np.maximum(bias, 1e-6)
    return raw, bias, corrected


def compute_window_stats(
    corrected: np.ndarray, raw: np.ndarray, min_valid: int = 5000
) -> Optional[Tuple[float, float, int]]:
    """Compute (mu, sigma, n_valid) over the brain-tissue voxels of a window.

    Brain voxels are defined as ``raw > 0`` (background in LSFM is exactly
    zero after the resampling pipeline). Returns None if not enough valid
    voxels are present (boundary windows).
    """
    valid = raw > 0
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return None
    vals = corrected[valid].astype(np.float32)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma < 1e-6:
        return None
    return mu, sigma, n_valid


# ---------------------------------------------------------------------------
# Disease-side patch extraction + segmentation
# ---------------------------------------------------------------------------

def normalize_window(window: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return ((window.astype(np.float32) - mu) / max(sigma, 1e-6)).astype(np.float32)


def central_crop(window: np.ndarray, crop: int = CROP_SIZE) -> np.ndarray:
    z, y, x = window.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    half = crop // 2
    return window[cz - half : cz + half, cy - half : cy + half, cx - half : cx + half].copy()


def segment_patch(
    normalized: np.ndarray,
    sigma_blur: float = GAUSSIAN_SIGMA,
    min_voxels: int = MIN_OBJECT_VOXELS,
) -> Dict[str, Dict]:
    """Run blur -> multi-Otsu (k=2 and k=3) -> CC -> size filter.

    Returns a dict keyed by 'otsu2' and 'otsu3' with mask/labels/n_components.
    """
    blurred = gaussian_filter(normalized.astype(np.float32), sigma=sigma_blur)
    out: Dict[str, Dict] = {}
    for k in (2, 3):
        try:
            thresholds = threshold_multiotsu(blurred, classes=k)
            mask = blurred > thresholds[-1]
        except ValueError:
            # Histogram is degenerate (e.g. uniform background patch).
            mask = np.zeros_like(blurred, dtype=bool)
        if mask.any():
            labels = cc3d.connected_components(
                mask.astype(np.uint8), connectivity=CC_CONNECTIVITY
            )
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0  # ignore background
            keep_ids = np.where(sizes >= min_voxels)[0]
            if keep_ids.size == 0:
                clean_mask = np.zeros_like(mask, dtype=np.uint8)
                clean_labels = np.zeros_like(labels, dtype=np.uint32)
                n_components = 0
            else:
                keep_set = set(int(i) for i in keep_ids)
                relabel = np.zeros(labels.max() + 1, dtype=np.uint32)
                next_id = 1
                for old in keep_ids:
                    relabel[old] = next_id
                    next_id += 1
                clean_labels = relabel[labels]
                clean_mask = (clean_labels > 0).astype(np.uint8)
                n_components = int(keep_set.__len__())
        else:
            clean_mask = np.zeros_like(mask, dtype=np.uint8)
            clean_labels = np.zeros_like(blurred, dtype=np.uint32)
            n_components = 0
        out[f"otsu{k}"] = {
            "mask": clean_mask,
            "labels": clean_labels,
            "n_components": n_components,
        }
    out["blurred"] = blurred
    return out


# ---------------------------------------------------------------------------
# QC figure
# ---------------------------------------------------------------------------

def _percentile_clip(
    arr: np.ndarray, low: float = 5.0, high: float = 99.5, ignore_zeros: bool = True
) -> Tuple[float, float]:
    """Robust percentile clip. Ignores exact-zero background by default so the
    contrast isn't dragged down by the resampling-pad voxels surrounding LSFM
    brains.
    """
    finite = arr[np.isfinite(arr)]
    if ignore_zeros:
        finite = finite[finite > 0]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def render_qc(
    raw_crop: np.ndarray,
    norm_crop: np.ndarray,
    seg_results: Dict[str, Dict],
    title: str,
    out_path: Path,
):
    """5×3 QC figure: raw / normalized / otsu2 overlay / otsu3 overlay / CC colormap."""
    z = raw_crop.shape[0]
    mid = z // 2
    slice_idx = [max(0, mid - 10), mid, min(z - 1, mid + 10)]

    # Robust contrast: percentile over nonzero brain tissue only.
    raw_lo, raw_hi = _percentile_clip(raw_crop, low=5.0, high=99.5, ignore_zeros=True)
    # For the normalized z-score, we DON'T ignore zeros (the normalized field
    # legitimately has values near zero), but we use a wider range to expose
    # tail brightness from plaques.
    n_valid = norm_crop[np.isfinite(norm_crop)]
    if n_valid.size == 0:
        norm_lo, norm_hi = (-3.0, 6.0)
    else:
        norm_lo = float(np.percentile(n_valid, 2))
        norm_hi = float(np.percentile(n_valid, 99.7))
        # Always show at least up to z=4 to keep cross-patch comparisons sensible
        norm_hi = max(norm_hi, 4.0)

    fig, axes = plt.subplots(5, 3, figsize=(13, 19))

    # Row 0: raw disease
    for j, sl in enumerate(slice_idx):
        ax = axes[0, j]
        ax.imshow(raw_crop[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        ax.set_title(f"raw  z={sl}")
        ax.axis("off")

    # Row 1: normalized z-score
    for j, sl in enumerate(slice_idx):
        ax = axes[1, j]
        im = ax.imshow(
            norm_crop[sl], cmap="magma", vmin=norm_lo, vmax=norm_hi, origin="lower"
        )
        ax.set_title(f"WT-normalized  z={sl}")
        ax.axis("off")
    fig.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04, label="z-score")

    # Row 2: otsu2 mask overlay
    mask2 = seg_results["otsu2"]["mask"]
    n2 = seg_results["otsu2"]["n_components"]
    for j, sl in enumerate(slice_idx):
        ax = axes[2, j]
        ax.imshow(raw_crop[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        overlay = np.ma.masked_where(mask2[sl] == 0, mask2[sl])
        ax.imshow(overlay, cmap="autumn", alpha=0.55, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"otsu k=2 ({n2} comp)  z={sl}")
        ax.axis("off")

    # Row 3: otsu3 mask overlay
    mask3 = seg_results["otsu3"]["mask"]
    n3 = seg_results["otsu3"]["n_components"]
    for j, sl in enumerate(slice_idx):
        ax = axes[3, j]
        ax.imshow(raw_crop[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        overlay = np.ma.masked_where(mask3[sl] == 0, mask3[sl])
        ax.imshow(overlay, cmap="cool", alpha=0.55, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"otsu k=3 ({n3} comp)  z={sl}")
        ax.axis("off")

    # Row 4: connected components colormap (use otsu2 by default)
    labels2 = seg_results["otsu2"]["labels"].astype(np.float32)
    labels2[labels2 == 0] = np.nan
    for j, sl in enumerate(slice_idx):
        ax = axes[4, j]
        ax.imshow(raw_crop[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        ax.imshow(labels2[sl], cmap="tab20", alpha=0.7, origin="lower")
        ax.set_title(f"otsu2 components (color)  z={sl}")
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _entry_key(entry: Dict) -> Tuple[str, str, int]:
    """Stable identity key for an output manifest entry."""
    raw = entry.get("raw_path") or entry.get("normalized_path") or ""
    patch_num = -1
    if "_patch" in raw:
        try:
            patch_num = int(Path(raw).name.split("_patch")[1].split("_")[0])
        except Exception:
            patch_num = -1
    return (entry.get("dataset", ""), entry.get("subject_id", ""), patch_num)


def _failure_key(entry: Dict) -> Tuple[str, str, str]:
    """Stable identity key for a source-manifest patch (for failure tracking)."""
    return (
        entry.get("dataset", ""),
        entry.get("subject_id", ""),
        entry.get("region_group", ""),
    )


def load_existing_manifest(path: Path) -> List[Dict]:
    with open(path) as f:
        m = json.load(f)
    return m["patches"]


def split_disease_and_wt(patches: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    disease, wt = [], []
    for p in patches:
        if is_wt_subject(p["dataset"], p["subject_id"]):
            wt.append(p)
        else:
            disease.append(p)
    return disease, wt


def get_or_load_wt_refs(
    dataset_name: str,
    cache: Dict[Tuple[str, str], WTReference],
    stain: str = "Abeta",
) -> List[WTReference]:
    """Return list of WTReference for a batch.
    Vaccine batch returns the pooled WTs from all other batches.
    """
    if dataset_name == "mouse_app_vaccine_batch" or not WT_SUBJECTS.get(dataset_name):
        # pool from every other batch that has WT(s)
        refs = []
        for other_ds, subjs in WT_SUBJECTS.items():
            for s in subjs:
                refs.append(_get_or_load_one(other_ds, s, cache, stain))
        return refs

    return [_get_or_load_one(dataset_name, s, cache, stain) for s in WT_SUBJECTS[dataset_name]]


def _get_or_load_one(
    dataset_name: str,
    subject_id: str,
    cache: Dict[Tuple[str, str], WTReference],
    stain: str,
) -> WTReference:
    key = (dataset_name, subject_id)
    if key not in cache:
        cache[key] = load_wt_reference(dataset_name, subject_id, stain=stain)
    return cache[key]


def process_patch(
    patch: Dict,
    wt_refs: List[WTReference],
    disease_znimg: ZarrNii,
    stain: str,
    out_root: Path,
) -> Optional[Dict]:
    dataset_name = patch["dataset"]
    sub_id = patch["subject_id"]
    center_phys = patch["center_phys"]

    # Reuse existing center_vox when present (it was computed by build_finetune_data
    # via the same coordinate chain). Recompute defensively if missing.
    if "center_vox" in patch:
        center_vox = np.array(patch["center_vox"], dtype=np.float64)
    else:
        # Resolve disease-side coordinate chain
        sub_paths = find_subject_paths(dataset_name, sub_id)
        fullres_inv = np.array(
            ZarrNii.from_ome_zarr(sub_paths["fullres_zarr"], channels=[0]).affine.invert().matrix,
            dtype=np.float64,
        )
        with open(sub_paths["sidecar"]) as f:
            sc = json.load(f)["fullres_to_resampled"]["scale"]
        res_scale = np.array([sc["z"], sc["y"], sc["x"]], dtype=np.float64)
        center_vox = _phys_to_resampled_voxel(fullres_inv, res_scale, center_phys)

    # 1) Disease 256³ window
    raw_window, _ = extract_window_from_zarr(disease_znimg, center_vox, size=PATCH_WINDOW)

    # 2) WT mu/sigma at the same physical coord, pooled across WTs.
    # If the physical coordinate doesn't land inside a given WT brain
    # (different physical-space conventions across animals), we fall back
    # to that WT's whole-brain global mu/sigma so the patch isn't lost.
    mu_vals: List[float] = []
    var_vals: List[float] = []
    n_valid_vals: List[int] = []
    used_wts: List[str] = []
    fallback_count = 0
    for wt in wt_refs:
        wt_raw, wt_bias, wt_corr = extract_corrected_window(wt, center_phys, size=PATCH_WINDOW)
        stats = compute_window_stats(wt_corr, wt_raw)
        if stats is not None:
            mu, sigma, n_valid = stats
            used_wts.append(f"{wt.dataset_name}/{wt.subject_id}")
        else:
            # Fall back to whole-brain WT stats
            mu = wt.global_mu
            sigma = wt.global_sigma
            n_valid = -1  # sentinel = global fallback
            used_wts.append(f"{wt.dataset_name}/{wt.subject_id}[global]")
            fallback_count += 1
        mu_vals.append(float(mu))
        var_vals.append(float(sigma * sigma))
        n_valid_vals.append(int(n_valid))

    if not mu_vals:
        print(f"  SKIP {dataset_name}/{sub_id} {patch.get('region_group', '?')}: no WT references at all")
        return None

    mu = float(np.mean(mu_vals))
    sigma = float(np.sqrt(np.mean(var_vals)))  # pooled (mean of variances)

    # 3) Normalize the disease window with WT stats
    normalized_window = normalize_window(raw_window, mu, sigma)

    # 4) Crop center 128³
    raw_crop = central_crop(raw_window, CROP_SIZE)
    norm_crop = central_crop(normalized_window, CROP_SIZE)

    # 5) Segmentation pipeline on the normalized crop
    seg_results = segment_patch(norm_crop)

    # 6) Save outputs
    patch_idx = patch.get("nii_path", "")
    # Pull patch number from existing nii_path if present, else generate
    if patch_idx and "_patch" in patch_idx:
        # e.g. ".../sub-XXX_patch07_crop128.nii.gz"
        try:
            stem = Path(patch_idx).name
            num = stem.split("_patch")[1].split("_")[0]
            patch_num = int(num)
        except Exception:
            patch_num = 0
    else:
        patch_num = 0

    out_dir = out_root / dataset_name / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{sub_id}_patch{patch_num:02d}"

    affine_4um = np.diag([0.004, 0.004, 0.004, 1.0])
    raw_path = out_dir / f"{prefix}_crop128.nii.gz"
    norm_path = out_dir / f"{prefix}_normalized128.nii.gz"
    seg2_path = out_dir / f"{prefix}_seg_otsu2.nii.gz"
    seg3_path = out_dir / f"{prefix}_seg_otsu3.nii.gz"
    qc_path = out_dir / f"{prefix}_qc.png"
    meta_path = out_dir / f"{prefix}_meta.json"

    nib.save(nib.Nifti1Image(raw_crop.astype(np.float32), affine_4um), str(raw_path))
    nib.save(nib.Nifti1Image(norm_crop.astype(np.float32), affine_4um), str(norm_path))
    nib.save(
        nib.Nifti1Image(seg_results["otsu2"]["mask"].astype(np.uint8), affine_4um),
        str(seg2_path),
    )
    nib.save(
        nib.Nifti1Image(seg_results["otsu3"]["mask"].astype(np.uint8), affine_4um),
        str(seg3_path),
    )

    # 7) QC figure
    title = (
        f"{dataset_name} / {sub_id} / patch{patch_num:02d}  region={patch.get('region_group','?')}\n"
        f"WT refs: {', '.join(used_wts)}\n"
        f"mu_wt={mu:.1f}  sigma_wt={sigma:.1f}  | "
        f"otsu2={seg_results['otsu2']['n_components']} comp  "
        f"otsu3={seg_results['otsu3']['n_components']} comp"
    )
    render_qc(raw_crop, norm_crop, seg_results, title, qc_path)

    meta = {
        "subject_id": sub_id,
        "dataset": dataset_name,
        "region_group": patch.get("region_group"),
        "center_phys": list(center_phys),
        "center_vox": [float(v) for v in center_vox],
        "stain": stain,
        "stain_channel": int(patch.get("stain_channel", 0)),
        "wt_references": used_wts,
        "wt_per_ref_mu": mu_vals,
        "wt_per_ref_sigma": [float(np.sqrt(v)) for v in var_vals],
        "wt_per_ref_n_valid": n_valid_vals,
        "wt_fallback_count": int(fallback_count),
        "mu_wt": mu,
        "sigma_wt": sigma,
        "n_components_otsu2": seg_results["otsu2"]["n_components"],
        "n_components_otsu3": seg_results["otsu3"]["n_components"],
        "raw_path": str(raw_path.relative_to(REPO_ROOT)),
        "normalized_path": str(norm_path.relative_to(REPO_ROOT)),
        "seg_otsu2_path": str(seg2_path.relative_to(REPO_ROOT)),
        "seg_otsu3_path": str(seg3_path.relative_to(REPO_ROOT)),
        "qc_path": str(qc_path.relative_to(REPO_ROOT)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    fb_str = f" [{fallback_count}/{len(wt_refs)} WT fallback]" if fallback_count else ""
    print(
        f"  ok  {dataset_name}/{sub_id} patch{patch_num:02d} "
        f"mu={mu:.1f} sigma={sigma:.1f} k2={seg_results['otsu2']['n_components']} "
        f"k3={seg_results['otsu3']['n_components']}{fb_str}"
    )
    return meta


def rebuild_manifest_from_disk(stain: str = "Abeta") -> None:
    """Walk OUTPUT_ROOT and rebuild manifest.json from per-patch meta.json files.

    Useful if a partial run accidentally overwrote the manifest with fewer
    entries than the on-disk patches.
    """
    print(f"Rebuilding {OUTPUT_ROOT / 'manifest.json'} from per-patch meta.json files ...")
    patches_meta: List[Dict] = []
    for meta_p in sorted(OUTPUT_ROOT.rglob("*_meta.json")):
        # Skip the wt_references/ meta sidecars
        if meta_p.parent == WT_CACHE_DIR:
            continue
        if "wt_references" in meta_p.parts:
            continue
        try:
            with open(meta_p) as f:
                m = json.load(f)
            patches_meta.append(m)
        except Exception as e:
            print(f"  WARN: bad meta {meta_p}: {e}")

    # Determine excluded WT patches by re-loading source manifest
    src_patches = load_existing_manifest(EXISTING_MANIFEST)
    _, wt_patches = split_disease_and_wt(src_patches)

    out_manifest = {
        "config": {
            "stain": stain,
            "patch_window": PATCH_WINDOW,
            "crop_size": CROP_SIZE,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "min_object_voxels": MIN_OBJECT_VOXELS,
            "cc_connectivity": CC_CONNECTIVITY,
            "n4_base_level_cap": N4_BASE_LEVEL,
            "n4_extra_coarsen": N4_EXTRA_COARSEN,
            "n4_iterations": N4_ITERATIONS,
            "wt_subjects": WT_SUBJECTS,
            "source_manifest": str(EXISTING_MANIFEST.relative_to(REPO_ROOT)),
            "n_patches_in": len(src_patches),
            "n_excluded_wt": len(wt_patches),
            "n_disease_processed": len(patches_meta),
            "n_failed": 0,
            "rebuilt_from_disk": True,
        },
        "wt_excluded": [
            {k: p[k] for k in ("subject_id", "dataset", "region_group") if k in p}
            for p in wt_patches
        ],
        "patches": patches_meta,
        "failures": [],
    }
    out_path = OUTPUT_ROOT / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(out_manifest, f, indent=2)
    print(f"  rebuilt manifest with {len(patches_meta)} patches -> {out_path}")


def main(
    stain: str = "Abeta",
    limit: Optional[int] = None,
    only_subjects: Optional[List[str]] = None,
    only_datasets: Optional[List[str]] = None,
):
    print(f"\n{'='*70}")
    print(f"Building WT-normalized {stain} fine-tune data")
    print(f"  existing manifest: {EXISTING_MANIFEST}")
    print(f"  output:            {OUTPUT_ROOT}")
    print(f"{'='*70}\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    patches = load_existing_manifest(EXISTING_MANIFEST)
    disease_patches, wt_patches = split_disease_and_wt(patches)
    print(f"  total patches in source manifest: {len(patches)}")
    print(f"  disease (will process): {len(disease_patches)}")
    print(f"  wt-animal patches (excluded): {len(wt_patches)}")
    if wt_patches:
        for p in wt_patches:
            print(f"    - {p['dataset']}/{p['subject_id']} patch ({p.get('region_group','?')})")

    if only_subjects:
        before = len(disease_patches)
        disease_patches = [p for p in disease_patches if p["subject_id"] in only_subjects]
        print(f"  --subject filter: {before} -> {len(disease_patches)} (kept {only_subjects})")

    if only_datasets:
        before = len(disease_patches)
        disease_patches = [p for p in disease_patches if p["dataset"] in only_datasets]
        print(f"  --dataset filter: {before} -> {len(disease_patches)} (kept {only_datasets})")

    if limit is not None:
        disease_patches = disease_patches[:limit]
        print(f"  LIMIT: only processing first {limit} disease patch(es)")

    # Group by dataset for efficient WT loading
    by_ds: Dict[str, List[Dict]] = defaultdict(list)
    for p in disease_patches:
        by_ds[p["dataset"]].append(p)

    wt_cache: Dict[Tuple[str, str], WTReference] = {}
    out_entries: List[Dict] = []
    failures: List[Dict] = []

    for dataset_name in sorted(by_ds.keys()):
        ds_patches = by_ds[dataset_name]
        print(f"\n[{dataset_name}] {len(ds_patches)} patch(es)")
        wt_refs = get_or_load_wt_refs(dataset_name, wt_cache, stain=stain)
        print(f"  WT references: {[f'{r.dataset_name}/{r.subject_id}' for r in wt_refs]}")

        # Group patches by subject so we only open each disease zarr once
        by_subj: Dict[str, List[Dict]] = defaultdict(list)
        for p in ds_patches:
            by_subj[p["subject_id"]].append(p)

        for sub_id, sp in sorted(by_subj.items()):
            sub_paths = find_subject_paths(dataset_name, sub_id)
            if not sub_paths["resampled_zarr"]:
                print(f"  SKIP {sub_id}: no resampled zarr found")
                for p in sp:
                    failures.append({"reason": "no resampled zarr", **p})
                continue
            stain_ch = resolve_stain_channel(
                sub_paths["resampled_zarr"], stain, sub_paths["fullres_zarr"]
            )
            disease_znimg = ZarrNii.from_ome_zarr(
                sub_paths["resampled_zarr"], channels=[stain_ch]
            )

            for p in sp:
                try:
                    entry = process_patch(p, wt_refs, disease_znimg, stain, OUTPUT_ROOT)
                except Exception as e:
                    print(f"  FAIL {dataset_name}/{sub_id}: {e}")
                    failures.append({"reason": str(e), **p})
                    continue
                if entry is not None:
                    out_entries.append(entry)
                else:
                    failures.append({"reason": "no wt stats", **p})

    # Write output manifest. When running with --subject/--dataset filter we
    # MERGE into the existing manifest by (dataset, subject_id, patch_num) so
    # we never silently drop the rest of the previously-built patches.
    out_manifest_path = OUTPUT_ROOT / "manifest.json"
    is_partial_run = bool(only_subjects) or bool(only_datasets)

    merged_entries: List[Dict] = []
    merged_failures: List[Dict] = []
    if is_partial_run and out_manifest_path.exists():
        try:
            with open(out_manifest_path) as f:
                existing = json.load(f)
            new_keys = {_entry_key(e) for e in out_entries}
            for e in existing.get("patches", []):
                if _entry_key(e) not in new_keys:
                    merged_entries.append(e)
            re_keys = {_failure_key(p) for p in disease_patches}
            for fent in existing.get("failures", []):
                if _failure_key(fent) not in re_keys:
                    merged_failures.append(fent)
            print(
                f"  merge: kept {len(merged_entries)} pre-existing entries from manifest"
            )
        except Exception as e:
            print(f"  WARN: failed to merge with existing manifest: {e}")

    merged_entries.extend(out_entries)
    merged_failures.extend(failures)

    out_manifest = {
        "config": {
            "stain": stain,
            "patch_window": PATCH_WINDOW,
            "crop_size": CROP_SIZE,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "min_object_voxels": MIN_OBJECT_VOXELS,
            "cc_connectivity": CC_CONNECTIVITY,
            "n4_base_level_cap": N4_BASE_LEVEL,
            "n4_extra_coarsen": N4_EXTRA_COARSEN,
            "n4_iterations": N4_ITERATIONS,
            "wt_subjects": WT_SUBJECTS,
            "source_manifest": str(EXISTING_MANIFEST.relative_to(REPO_ROOT)),
            "n_patches_in": len(patches),
            "n_excluded_wt": len(wt_patches),
            "n_disease_processed": len(merged_entries),
            "n_failed": len(merged_failures),
            "is_partial_run": is_partial_run,
            "this_run_processed": len(out_entries),
            "this_run_failed": len(failures),
        },
        "wt_excluded": [
            {k: p[k] for k in ("subject_id", "dataset", "region_group") if k in p}
            for p in wt_patches
        ],
        "patches": merged_entries,
        "failures": merged_failures,
    }
    with open(out_manifest_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  This run: wrote {len(out_entries)} patches  failed {len(failures)}")
    if is_partial_run:
        print(f"  Merged with existing -> manifest now contains {len(merged_entries)} total")
    print(f"  Manifest: {out_manifest_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stain", default="Abeta")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only process the first N disease patches (smoke test)",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Restrict processing to one or more subject IDs (repeatable). "
             "Example: --subject sub-AS40F2 --subject sub-AS114M3",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Restrict processing to one or more dataset names (repeatable). "
             "Example: --dataset mouse_app_lecanemab_batch2",
    )
    parser.add_argument(
        "--rebuild-manifest-from-disk",
        action="store_true",
        help="Walk OUTPUT_ROOT and rebuild manifest.json from per-patch meta.json "
             "files. Use this if a partial run accidentally overwrote the manifest. "
             "Does NOT rerun the pipeline.",
    )
    args = parser.parse_args()
    if args.rebuild_manifest_from_disk:
        rebuild_manifest_from_disk(stain=args.stain)
    else:
        main(
            stain=args.stain,
            limit=args.limit,
            only_subjects=args.subject,
            only_datasets=args.dataset,
        )
