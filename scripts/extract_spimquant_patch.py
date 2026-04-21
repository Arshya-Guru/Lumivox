"""extract_spimquant_patch.py — pull SPIMquant's full-res Abeta mask for the
same 128³ region used by sub-AS114M3 patch00 in ft_normalized/Abeta/manifest.json
and downsample it to 4 µm isotropic, so it can be visually compared in napari
against the existing seg_otsu2 / seg_otsu3 / GOLD masks for that patch.

Pipeline:
  1. Find the patch entry in ft_normalized/Abeta/manifest.json by patch_id.
  2. Read the resampled-zarr JSON sidecar for that subject — it contains the
     scale factors mapping fullres voxel index -> resampled voxel index.
  3. Convert ``center_vox`` (manifest, resampled-4 µm space) back to full-res
     voxel coordinates by dividing by the scale per axis.
  4. Open the SPIMquant binary mask (level-0, full-res) with ZarrNii. The .ozx
     extension is just an OME-Zarr — ZarrNii.from_ome_zarr opens it directly.
  5. Verify the mask shape matches the full-res zarr shape (same voxel grid).
  6. Extract a region whose physical extent equals 128 voxels of 4 µm — i.e.
     a (128/scale_z, 128/scale_y, 128/scale_x) box in full-res voxels — centered
     on the converted center.
  7. Downsample the extracted box to 128³ with nearest-neighbor (binary mask).
  8. Save as a NIfTI with the same affine as the existing crop128.nii.gz.

Defaults are hardcoded for sub-AS114M3 patch00 in mouse_app_lecanemab_batch2.
Override via --patch / --spimquant-mask if you ever want to extract for a
different patch.

Usage:
    pixi run python scripts/extract_spimquant_patch.py
    pixi run python scripts/extract_spimquant_patch.py --patch sub-AS114M3_patch00
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom
from zarrnii import ZarrNii


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "ft_normalized" / "Abeta" / "manifest.json"

# $LS env var falls back to the prado lightsheet root we've been working with.
LS_ROOT = Path(os.environ.get("LS", "/nfs/trident3/lightsheet/prado"))

# Manifest dataset name -> on-disk directory name. The vaccine batch stored
# itself in the existing manifest as 'mouse_app_vaccine_batch' but actually
# lives at 'mouse_app_vaccine_batch1' on disk.
DATASET_DIR_OVERRIDES = {"mouse_app_vaccine_batch": "mouse_app_vaccine_batch1"}


def _ds_dir(dataset: str) -> str:
    return DATASET_DIR_OVERRIDES.get(dataset, dataset)

# Defaults: AS114M3 patch00, the patch the user wants to compare.
DEFAULT_PATCH_ID = "sub-AS114M3_patch00"
DEFAULT_SPIMQUANT_MASK = (
    LS_ROOT
    / "mouse_app_lecanemab_batch2"
    / "derivatives"
    / "spimquant-v0.6.0rc2_84a605e_ozx"
    / "sub-AS114M3"
    / "micr"
    / "sub-AS114M3_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-otsu+k3i2_mask.ozx"
)

CROP_SIZE = 128


# ---------------------------------------------------------------------------
# Manifest + sidecar lookup
# ---------------------------------------------------------------------------

def _patch_id(entry: dict) -> str:
    raw = entry.get("raw_path", "")
    if not raw:
        return f"{entry.get('subject_id','unknown')}_unknown"
    name = Path(raw).name
    return name.split("_crop")[0] if "_crop" in name else name.split(".")[0]


def find_patch_entry(patch_id: str) -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    for p in manifest["patches"]:
        if _patch_id(p) == patch_id:
            return p
    sys.exit(
        f"could not find {patch_id} in {MANIFEST_PATH}. "
        f"available patches: {[_patch_id(p) for p in manifest['patches'][:5]]}..."
    )


def find_resampled_zarr(dataset: str, subject_id: str) -> Path:
    """Locate the *_res-4um_SPIM.ome.zarr for a subject under $LS."""
    micr = LS_ROOT / _ds_dir(dataset) / "bids" / "derivatives" / "resampled" / subject_id / "micr"
    if not micr.exists():
        sys.exit(f"resampled micr dir not found: {micr}")
    candidates = sorted(micr.glob("*_res-4um_SPIM.ome.zarr"))
    if not candidates:
        sys.exit(f"no res-4um zarr in {micr}")
    return candidates[0]


def find_fullres_zarr(dataset: str, subject_id: str) -> Path:
    """Locate the full-res *_SPIM.ome.zarr (used to verify mask shape match)."""
    micr = LS_ROOT / _ds_dir(dataset) / "bids" / subject_id / "micr"
    if not micr.exists():
        sys.exit(f"fullres micr dir not found: {micr}")
    candidates = sorted(
        c for c in micr.glob("*_SPIM.ome.zarr")
        if "45deg" not in c.name and "90deg" not in c.name
    )
    if not candidates:
        sys.exit(f"no fullres zarr in {micr}")
    return candidates[0]


def read_sidecar_scale(resampled_zarr: Path) -> np.ndarray:
    """Return the (z, y, x) scale factors with the convention
    ``voxel_resampled = voxel_fullres * scale``.
    """
    sidecar = Path(str(resampled_zarr) + ".json")
    if not sidecar.exists():
        sys.exit(f"sidecar not found: {sidecar}")
    with open(sidecar) as f:
        sc = json.load(f)["fullres_to_resampled"]["scale"]
    return np.array([sc["z"], sc["y"], sc["x"]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Crop extraction + downsample
# ---------------------------------------------------------------------------

def extract_fullres_box(
    mask_darr,
    center_fullres: np.ndarray,
    half_fullres: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    """Extract an axis-aligned box from a (Z, Y, X) dask array, padding with
    zeros at the boundary. Returns the box AND the bounds for logging.
    """
    z0 = int(round(center_fullres[0] - half_fullres[0]))
    z1 = int(round(center_fullres[0] + half_fullres[0]))
    y0 = int(round(center_fullres[1] - half_fullres[1]))
    y1 = int(round(center_fullres[1] + half_fullres[1]))
    x0 = int(round(center_fullres[2] - half_fullres[2]))
    x1 = int(round(center_fullres[2] + half_fullres[2]))

    Z, Y, X = mask_darr.shape
    sz0, sz1 = max(0, z0), min(Z, z1)
    sy0, sy1 = max(0, y0), min(Y, y1)
    sx0, sx1 = max(0, x0), min(X, x1)

    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        slab = mask_darr[sz0:sz1, sy0:sy1, sx0:sx1].compute()
        out[
            sz0 - z0 : sz0 - z0 + slab.shape[0],
            sy0 - y0 : sy0 - y0 + slab.shape[1],
            sx0 - x0 : sx0 - x0 + slab.shape[2],
        ] = (np.asarray(slab) > 0).astype(np.uint8)

    return out, (z0, z1, y0, y1, x0, x1)


def downsample_to_4um(box: np.ndarray, mode: str = "nn") -> np.ndarray:
    """Resample an arbitrary-shape binary box to 128³.

    The box has been carved so its physical extent is exactly 128 voxels of
    4 µm per axis. Two modes:

      mode="nn"   - nearest-neighbor via scipy.ndimage.zoom(order=0). Matches
                    the original spec. Cheap but loses isolated positive voxels
                    when the source is very sparse (any positive that doesn't
                    land on a sample point disappears).
      mode="max"  - block-max pooling via index buckets. Any output voxel that
                    contains at least one positive source voxel becomes 1.
                    Use this when SPIMquant's mask is sparse (e.g. dim
                    hippocampal patches) and you need *something* to inspect.
    """
    if mode not in ("nn", "max"):
        raise ValueError(f"unknown downsample mode: {mode}")

    if mode == "nn":
        zoom_factors = (
            CROP_SIZE / box.shape[0],
            CROP_SIZE / box.shape[1],
            CROP_SIZE / box.shape[2],
        )
        out = ndi_zoom(box.astype(np.float32), zoom_factors, order=0)
        out = (out > 0.5).astype(np.uint8)
    else:
        # Block-max pooling: assign each source voxel to an output bin and OR
        # everything in the same bin together. Works for arbitrary downsample
        # ratios (no need for the source dims to be a multiple of 128).
        binary = (box > 0).astype(np.uint8)
        out = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        nz = np.argwhere(binary > 0)
        if nz.size:
            sz = CROP_SIZE / box.shape[0]
            sy = CROP_SIZE / box.shape[1]
            sx = CROP_SIZE / box.shape[2]
            iz = np.minimum((nz[:, 0] * sz).astype(np.int64), CROP_SIZE - 1)
            iy = np.minimum((nz[:, 1] * sy).astype(np.int64), CROP_SIZE - 1)
            ix = np.minimum((nz[:, 2] * sx).astype(np.int64), CROP_SIZE - 1)
            out[iz, iy, ix] = 1

    if out.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
        padded = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        zs = min(out.shape[0], CROP_SIZE)
        ys = min(out.shape[1], CROP_SIZE)
        xs = min(out.shape[2], CROP_SIZE)
        padded[:zs, :ys, :xs] = out[:zs, :ys, :xs]
        out = padded

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", default=DEFAULT_PATCH_ID,
                   help=f"patch ID (default: {DEFAULT_PATCH_ID})")
    p.add_argument("--spimquant-mask", default=str(DEFAULT_SPIMQUANT_MASK),
                   help="path to the SPIMquant level-0 mask .ozx")
    p.add_argument("--out", default=None,
                   help="output NIfTI path (default: alongside crop128.nii.gz)")
    p.add_argument("--downsample", choices=["nn", "max"], default="nn",
                   help="downsample mode for fullres -> 128³ 4 um. nn = "
                        "nearest-neighbor (spec default; loses isolated voxels). "
                        "max = block-max pool (preserves any positive voxel; "
                        "useful when SPIMquant's mask is sparse).")
    args = p.parse_args()

    patch_id = args.patch
    spimquant_mask_path = Path(args.spimquant_mask)

    # 1. Manifest entry
    entry = find_patch_entry(patch_id)
    print(f"\n=== {patch_id} ===")
    print(f"  dataset:    {entry['dataset']}")
    print(f"  subject:    {entry['subject_id']}")
    print(f"  region:     {entry.get('region_group','?')}")
    center_vox_resampled = np.array(entry["center_vox"], dtype=np.float64)
    print(f"  center_vox (resampled 4 um): {center_vox_resampled}")

    # 2. Resampled zarr -> sidecar -> scale factors
    resampled_zarr = find_resampled_zarr(entry["dataset"], entry["subject_id"])
    print(f"  resampled zarr: {resampled_zarr}")
    scale = read_sidecar_scale(resampled_zarr)
    print(f"  scale (z,y,x): {scale}  (resampled = fullres * scale)")

    # 3. Convert center: resampled -> fullres voxel
    center_fullres = center_vox_resampled / scale
    print(f"  center_vox (fullres):       {center_fullres}")

    # 4. Open SPIMquant mask
    if not spimquant_mask_path.exists():
        sys.exit(f"SPIMquant mask not found: {spimquant_mask_path}")
    print(f"\n  opening SPIMquant mask: {spimquant_mask_path}")
    znimg_mask = ZarrNii.from_ome_zarr(str(spimquant_mask_path), channels=[0])
    mask_darr = znimg_mask.darr[0]  # drop channel
    print(f"  spimquant mask shape: {mask_darr.shape}, dtype: {mask_darr.dtype}")

    # 5. Verify same voxel grid as the full-res zarr
    fullres_zarr = find_fullres_zarr(entry["dataset"], entry["subject_id"])
    znimg_full = ZarrNii.from_ome_zarr(str(fullres_zarr), channels=[0])
    fullres_shape = znimg_full.darr.shape[1:]
    if tuple(mask_darr.shape) != tuple(fullres_shape):
        sys.exit(
            f"shape mismatch: spimquant mask {mask_darr.shape} != fullres zarr {fullres_shape}"
        )
    if not np.allclose(np.asarray(znimg_mask.affine.matrix),
                       np.asarray(znimg_full.affine.matrix)):
        print("  WARN: spimquant affine differs from fullres zarr affine")
        print(f"    spimquant:\n{znimg_mask.affine.matrix}")
        print(f"    fullres:\n{znimg_full.affine.matrix}")
    else:
        print(f"  affines match fullres zarr — same voxel grid")

    # 6. Crop window in fullres voxels (physical extent = 128 voxels at 4 um)
    half_fullres = (CROP_SIZE / 2.0) / scale
    print(f"  half-window in fullres voxels (z,y,x): {half_fullres}")

    box, (z0, z1, y0, y1, x0, x1) = extract_fullres_box(
        mask_darr, center_fullres, half_fullres
    )
    print(f"  fullres box bounds:")
    print(f"    z: [{z0}, {z1}] ({z1 - z0} vox)")
    print(f"    y: [{y0}, {y1}] ({y1 - y0} vox)")
    print(f"    x: [{x0}, {x1}] ({x1 - x0} vox)")
    fullres_pos = int((box > 0).sum())
    fullres_density = fullres_pos / box.size
    print(f"  fullres box shape={box.shape} positive={fullres_pos} ({fullres_density:.4%})")

    # 7. Downsample to 4 um (128³)
    crop_4um = downsample_to_4um(box, mode=args.downsample)
    crop_pos = int(crop_4um.sum())
    print(f"  downsampled ({args.downsample}) shape={crop_4um.shape} positive={crop_pos}")
    if args.downsample == "nn" and fullres_pos > 0 and crop_pos == 0:
        print(
            f"  NOTE: nearest-neighbor downsample lost all {fullres_pos} positive voxels."
            f" Try --downsample max to preserve them."
        )

    # 8. Save with the same affine as the existing crop128.nii.gz
    raw_path = REPO_ROOT / entry["raw_path"]
    if not raw_path.exists():
        sys.exit(f"existing crop128 NIfTI not found: {raw_path}")
    ref = nib.load(str(raw_path))

    if args.out is None:
        suffix = "_spimquant_mask" if args.downsample == "nn" else "_spimquant_mask_maxpool"
        out_path = raw_path.parent / f"{patch_id}{suffix}.nii.gz"
    else:
        out_path = Path(args.out)

    nib.save(nib.Nifti1Image(crop_4um, ref.affine), str(out_path))
    print(f"\n  saved: {out_path}")
    print(
        f"\nLoad in napari alongside the existing layers:"
        f"\n  bash {raw_path.parent / f'view_{patch_id}.sh'}"
        f"\n(then File -> Open and add {out_path.name})\n"
    )


if __name__ == "__main__":
    main()
