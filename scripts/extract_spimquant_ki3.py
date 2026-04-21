"""extract_spimquant_ki3.py — Extract SPIMquant masks for KI3 patches.

KI3 batches store SPIMquant masks as ``.patches`` directories containing
per-region NIfTI tiles (256³ each at full-res) rather than full-volume
``.ozx`` archives. This script assembles the tiles that overlap each
patch's physical bounding box into a single 128³ 4 µm mask.

Usage:
    pixi run python scripts/extract_spimquant_ki3.py              # all ki3 patches
    pixi run python scripts/extract_spimquant_ki3.py --patch sub-AS134F1_patch00
    pixi run python scripts/extract_spimquant_ki3.py --batch ki3_batch1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom

REPO_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = REPO_ROOT / "ft_normalized" / "Abeta"
LS_ROOT = Path("/nfs/trident3/lightsheet/prado")
CROP_SIZE = 128

# SPIMquant version per KI3 batch and the mask description pattern.
# acq-imaris4x ONLY (not 4x166).
KI3_SPIMQUANT = {
    "mouse_app_lecanemab_ki3_batch1": {
        "spimquant": "spimquant_v0.5.0-alpha1",
        "mask_glob": "*_acq-imaris4x_stain-Abeta_*_mask.patches",
    },
    "mouse_app_lecanemab_ki3_batch2": {
        "spimquant": "spimquant_c270a40_atropos",
        "mask_glob": "*_acq-imaris4x_stain-Abeta_*_mask.patches",
    },
    "mouse_app_lecanemab_ki3_batch3": {
        "spimquant": "spimquant_c270a40_atropos",
        "mask_glob": "*_acq-imaris4x_stain-Abeta_*_mask.patches",
    },
}


def find_mask_patches_dir(dataset: str, subject_id: str) -> Optional[Path]:
    """Locate the .patches directory for a KI3 subject (acq-imaris4x only)."""
    cfg = KI3_SPIMQUANT.get(dataset)
    if cfg is None:
        return None
    micr = LS_ROOT / dataset / "derivatives" / cfg["spimquant"] / subject_id / "micr"
    if not micr.exists():
        return None
    candidates = sorted(micr.glob(cfg["mask_glob"]))
    # Exclude 4x166
    candidates = [c for c in candidates if "4x166" not in c.name]
    return candidates[0] if candidates else None


def find_resampled_sidecar(dataset: str, subject_id: str) -> Optional[Path]:
    micr = LS_ROOT / dataset / "bids" / "derivatives" / "resampled" / subject_id / "micr"
    if not micr.exists():
        return None
    candidates = sorted(micr.glob("*_res-4um_SPIM.ome.zarr.json"))
    candidates = [c for c in candidates if "4x166" not in c.name and "45deg" not in c.name]
    return candidates[0] if candidates else None


def load_tiles(patches_dir: Path) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
    """Load all tile headers (affine + shape) from a .patches directory.

    Returns list of (affine_4x4, inv_affine_4x4, shape) per tile — data is
    NOT loaded yet.
    """
    tiles = sorted(patches_dir.glob("*.nii*"))
    out = []
    for t in tiles:
        img = nib.load(str(t))
        out.append((np.array(img.affine, dtype=np.float64), t, img.shape))
    return out


def assemble_tiles(
    tiles: List[Tuple[np.ndarray, Path, Tuple[int, ...]]],
    center_phys: np.ndarray,
    crop_phys_mm: float,  # physical half-extent in mm per axis
) -> np.ndarray:
    """Assemble overlapping tiles into one volume covering the crop's physical
    bounding box. Returns a uint8 volume in *physical-mm* grid at the tile's
    native voxel resolution, shaped to exactly cover the crop bbox.
    """
    # Crop bounding box in physical mm
    lo = center_phys - crop_phys_mm
    hi = center_phys + crop_phys_mm

    # We need a common voxel grid. Use the first tile's voxel sizes as reference.
    ref_affine = tiles[0][0]
    voxel_sizes = np.abs(np.diag(ref_affine)[:3])  # (dx, dy, dz) in mm

    # Output volume dimensions (number of voxels spanning the crop bbox)
    out_shape = np.ceil((hi - lo) / voxel_sizes).astype(int)
    out = np.zeros(tuple(out_shape), dtype=np.uint8)

    # Build an affine for the output volume: maps voxel (0,0,0) -> lo corner,
    # with the same voxel sizes + orientation sign as the tiles.
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = ref_affine[i, i]  # preserve sign
    out_affine[:3, 3] = lo  # origin at lo corner

    out_inv = np.linalg.inv(out_affine)

    n_pasted = 0
    for tile_affine, tile_path, tile_shape in tiles:
        # Tile bounding box in physical mm
        tile_origin = tile_affine[:3, 3]
        tile_vox_sizes = np.abs(np.diag(tile_affine)[:3])
        tile_end = tile_origin.copy()
        for i in range(3):
            tile_end[i] = tile_origin[i] + tile_affine[i, i] * tile_shape[i]
        tile_lo = np.minimum(tile_origin, tile_end)
        tile_hi = np.maximum(tile_origin, tile_end)

        # Check overlap with crop bbox
        overlap_lo = np.maximum(lo, tile_lo)
        overlap_hi = np.minimum(hi, tile_hi)
        if np.any(overlap_hi <= overlap_lo):
            continue

        # Load tile data
        tile_data = nib.load(str(tile_path)).get_fdata().astype(np.uint8)
        tile_inv = np.linalg.inv(tile_affine)

        # Map overlap corners to tile voxels and output voxels
        for iz in range(out_shape[0]):
            phys_z = out_affine[0, 0] * iz + out_affine[0, 3]
            # Check if this z is within the tile's z range
            tz_f = (phys_z - tile_affine[0, 3]) / tile_affine[0, 0]
            tz = int(round(tz_f))
            if tz < 0 or tz >= tile_shape[0]:
                continue
            for iy in range(out_shape[1]):
                phys_y = out_affine[1, 1] * iy + out_affine[1, 3]
                ty_f = (phys_y - tile_affine[1, 3]) / tile_affine[1, 1]
                ty = int(round(ty_f))
                if ty < 0 or ty >= tile_shape[1]:
                    continue
                for ix in range(out_shape[2]):
                    phys_x = out_affine[2, 2] * ix + out_affine[2, 3]
                    tx_f = (phys_x - tile_affine[2, 3]) / tile_affine[2, 2]
                    tx = int(round(tx_f))
                    if tx < 0 or tx >= tile_shape[2]:
                        continue
                    if tile_data[tz, ty, tx] > 0:
                        out[iz, iy, ix] = 1
        n_pasted += 1

    return out, n_pasted


def assemble_tiles_fast(
    tiles: List[Tuple[np.ndarray, Path, Tuple[int, ...]]],
    center_phys: np.ndarray,
    crop_phys_mm: float,
) -> Tuple[np.ndarray, int]:
    """Vectorized version: assemble overlapping tiles using array slicing."""
    lo = center_phys - crop_phys_mm
    hi = center_phys + crop_phys_mm

    ref_affine = tiles[0][0]
    voxel_sizes = np.abs(np.diag(ref_affine)[:3])
    out_shape = np.ceil((hi - lo) / voxel_sizes).astype(int)
    out = np.zeros(tuple(out_shape), dtype=np.uint8)

    # Output affine: same sign conventions as tiles, origin at lo
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = ref_affine[i, i]
    out_affine[:3, 3] = lo

    n_pasted = 0
    for tile_affine, tile_path, tile_shape in tiles:
        tile_origin = tile_affine[:3, 3]
        tile_end = np.array([
            tile_origin[i] + tile_affine[i, i] * tile_shape[i]
            for i in range(3)
        ])
        tile_lo = np.minimum(tile_origin, tile_end)
        tile_hi = np.maximum(tile_origin, tile_end)

        overlap_lo = np.maximum(lo, tile_lo)
        overlap_hi = np.minimum(hi, tile_hi)
        if np.any(overlap_hi <= overlap_lo + 1e-9):
            continue

        # Map overlap region to tile voxel indices
        tile_data = nib.load(str(tile_path)).get_fdata().astype(np.uint8)

        # For each axis: compute the voxel range in the tile and in the output
        slices_tile = []
        slices_out = []
        for i in range(3):
            step = tile_affine[i, i]
            origin = tile_origin[i]
            # tile voxel index for overlap_lo and overlap_hi
            if step > 0:
                t0 = max(0, int(np.floor((overlap_lo[i] - origin) / step)))
                t1 = min(tile_shape[i], int(np.ceil((overlap_hi[i] - origin) / step)))
            else:
                t0 = max(0, int(np.floor((overlap_hi[i] - origin) / step)))
                t1 = min(tile_shape[i], int(np.ceil((overlap_lo[i] - origin) / step)))
            slices_tile.append(slice(t0, t1))

            # output voxel index for the same physical range
            o_step = out_affine[i, i]
            o_origin = out_affine[i, 3]
            phys_start = origin + step * t0
            phys_end = origin + step * t1
            if o_step > 0:
                o0 = max(0, int(np.floor((min(phys_start, phys_end) - o_origin) / o_step)))
                o1 = min(out_shape[i], o0 + (t1 - t0))
            else:
                o0 = max(0, int(np.floor((max(phys_start, phys_end) - o_origin) / o_step)))
                o1 = min(out_shape[i], o0 + (t1 - t0))
            slices_out.append(slice(o0, o1))

        # Extract overlapping region from tile
        tile_chunk = tile_data[slices_tile[0], slices_tile[1], slices_tile[2]]
        # Paste into output (OR to handle overlapping tiles)
        oz = slices_out[0]
        oy = slices_out[1]
        ox = slices_out[2]
        # Handle size mismatches from rounding
        sz = min(oz.stop - oz.start, tile_chunk.shape[0])
        sy = min(oy.stop - oy.start, tile_chunk.shape[1])
        sx = min(ox.stop - ox.start, tile_chunk.shape[2])
        if sz > 0 and sy > 0 and sx > 0:
            out[oz.start:oz.start+sz, oy.start:oy.start+sy, ox.start:ox.start+sx] |= (
                tile_chunk[:sz, :sy, :sx] > 0
            ).astype(np.uint8)
            n_pasted += 1

    return out, n_pasted


def downsample_maxpool(box: np.ndarray, target: int = CROP_SIZE) -> np.ndarray:
    """Block-max pool a binary volume to target³."""
    binary = (box > 0).astype(np.uint8)
    out = np.zeros((target, target, target), dtype=np.uint8)
    nz = np.argwhere(binary > 0)
    if nz.size:
        sz = target / box.shape[0]
        sy = target / box.shape[1]
        sx = target / box.shape[2]
        iz = np.minimum((nz[:, 0] * sz).astype(np.int64), target - 1)
        iy = np.minimum((nz[:, 1] * sy).astype(np.int64), target - 1)
        ix = np.minimum((nz[:, 2] * sx).astype(np.int64), target - 1)
        out[iz, iy, ix] = 1
    return out


def process_patch(meta_path: Path, force: bool = False) -> Optional[str]:
    patch_id = meta_path.name.replace("_meta.json", "")
    out_path = meta_path.parent / f"{patch_id}_spimquant_mask_maxpool.nii.gz"
    if out_path.exists() and not force:
        return None  # skip

    with open(meta_path) as f:
        meta = json.load(f)

    dataset = meta["dataset"]
    subject_id = meta["subject_id"]
    center_phys = np.array(meta["center_phys"], dtype=np.float64)

    # Find mask.patches directory
    patches_dir = find_mask_patches_dir(dataset, subject_id)
    if patches_dir is None:
        return f"SKIP {patch_id}: no mask.patches found for {dataset}/{subject_id}"

    # Physical crop half-extent = (CROP_SIZE/2) * 4um = 0.256 mm
    crop_half_mm = (CROP_SIZE / 2) * 0.004  # mm

    # Load tile headers
    tiles = load_tiles(patches_dir)

    # Assemble overlapping tiles
    assembled, n_tiles = assemble_tiles_fast(tiles, center_phys, crop_half_mm)
    pos_fullres = int((assembled > 0).sum())

    # Downsample to 128³ via max-pool
    crop_4um = downsample_maxpool(assembled)
    pos_4um = int(crop_4um.sum())

    # Save with same affine as the existing crop128
    ref_nii = meta_path.parent / f"{patch_id}_crop128.nii.gz"
    if not ref_nii.exists():
        return f"SKIP {patch_id}: no crop128.nii.gz"
    ref = nib.load(str(ref_nii))
    nib.save(nib.Nifti1Image(crop_4um, ref.affine), str(out_path))

    return (
        f"  {patch_id}  {meta.get('region_group','?'):<13} "
        f"tiles={n_tiles}  fullres={pos_fullres:>8}  4um={pos_4um:>6}  -> {out_path.name}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", default=None)
    p.add_argument("--batch", default=None, help="e.g. ki3_batch1")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # Find all ki3 meta.json files
    metas = []
    for batch in KI3_SPIMQUANT:
        if args.batch and args.batch not in batch:
            continue
        for mp in sorted((FT_DIR / batch).rglob("*_meta.json")):
            pid = mp.name.replace("_meta.json", "")
            if args.patch and pid != args.patch:
                continue
            metas.append(mp)

    if not metas:
        sys.exit("no ki3 patches found")

    print(f"processing {len(metas)} ki3 patches ...\n")
    n_done = 0
    n_skip = 0
    for mp in metas:
        result = process_patch(mp, force=args.force)
        if result is None:
            n_skip += 1
        elif result.startswith("SKIP"):
            print(result)
        else:
            print(result)
            n_done += 1

    print(f"\ndone. extracted={n_done}  skipped={n_skip}")


if __name__ == "__main__":
    main()
