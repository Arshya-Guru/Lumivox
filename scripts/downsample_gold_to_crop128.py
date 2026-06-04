"""downsample_gold_to_crop128.py — collapse a full-res seg_gold mask back to
the 128^3 4 um crop space using block max-pool.

For each input voxel (zi, yi, xi) with value v > 0, set the corresponding
output voxel (zo, yo, xo) = floor((zi, yi, xi) * (128 / fullres_shape)) to
max(out, v). Multi-label aware (np.maximum.at), so label IDs are preserved.
Mirrors the maxpool used by extract_spimquant_patch.py.

Usage:
    # default: input ends in "_fullres", output strips it, ref is the sibling crop128
    pixi run python scripts/downsample_gold_to_crop128.py \\
        ft_normalized/Abeta/.../sub-X_patch01_seg_gold_fullres.nii.gz

    # explicit:
    pixi run python scripts/downsample_gold_to_crop128.py <in> \\
        --output <out.nii.gz> --ref <crop128.nii.gz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

CROP_SIZE = 128


def maxpool_to_target(box: np.ndarray, target: int = CROP_SIZE) -> np.ndarray:
    """Block max-pool an integer-labeled volume to (target, target, target).
    Each nonzero input voxel maps to one output voxel; ties resolved by max."""
    out = np.zeros((target, target, target), dtype=box.dtype)
    nz = np.argwhere(box > 0)
    if nz.size == 0:
        return out
    sz = target / box.shape[0]
    sy = target / box.shape[1]
    sx = target / box.shape[2]
    iz = np.minimum((nz[:, 0] * sz).astype(np.int64), target - 1)
    iy = np.minimum((nz[:, 1] * sy).astype(np.int64), target - 1)
    ix = np.minimum((nz[:, 2] * sx).astype(np.int64), target - 1)
    vals = box[nz[:, 0], nz[:, 1], nz[:, 2]]
    np.maximum.at(out, (iz, iy, ix), vals)
    return out


def _derive_output(in_path: Path) -> Path:
    if "_fullres" not in in_path.name:
        sys.exit(
            f"can't derive --output from {in_path.name} (no '_fullres' in name). "
            "pass --output explicitly."
        )
    return in_path.parent / in_path.name.replace("_fullres", "")


def _derive_ref(in_path: Path) -> Path:
    # Strip known mask/fullres markers from the basename in either order
    # (sub-X_patch01_seg_gold_fullres or sub-X_patch01_fullres_seg_gold), then
    # form <patch_id>_crop128.nii.gz next to the input.
    name = in_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = name
    for marker in ("_seg_gold", "_fullres", "_gold"):
        stem = stem.replace(marker, "")
    if not stem:
        sys.exit(f"can't derive --ref crop128 from {in_path.name}. pass --ref explicitly.")
    return in_path.parent / f"{stem}_crop128.nii.gz"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="full-res seg_gold NIfTI")
    p.add_argument("--output", help="output NIfTI (default: strip '_fullres' from input name)")
    p.add_argument("--ref", help="reference 128^3 crop NIfTI for affine (default: sibling *_crop128.nii.gz)")
    p.add_argument("--size", type=int, default=CROP_SIZE, help="target cube size (default 128)")
    p.add_argument("--force", action="store_true", help="overwrite output if it exists")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    out_path = Path(args.output) if args.output else _derive_output(in_path)
    ref_path = Path(args.ref) if args.ref else _derive_ref(in_path)
    if not ref_path.exists():
        sys.exit(f"reference crop128 not found: {ref_path} (pass --ref to override)")
    if out_path.exists() and not args.force:
        sys.exit(f"output exists (use --force to overwrite): {out_path}")

    img = nib.load(str(in_path))
    box = np.asarray(img.dataobj)
    if box.ndim != 3:
        sys.exit(f"expected 3D volume, got shape {box.shape}")

    # work in int64 for safe np.maximum.at; cast back to a compact int dtype on save
    box_i = box.astype(np.int64)
    pre_pos = int((box_i > 0).sum())
    pre_max = int(box_i.max())

    out = maxpool_to_target(box_i, target=args.size)
    post_pos = int((out > 0).sum())
    post_max = int(out.max())

    out_dtype = np.uint8 if post_max <= 255 else np.uint16
    ref = nib.load(str(ref_path))
    nib.save(nib.Nifti1Image(out.astype(out_dtype), ref.affine), str(out_path))

    print(f"input:  {in_path}")
    print(f"  shape={box.shape}  nnz={pre_pos}  max={pre_max}")
    print(f"output: {out_path}")
    print(f"  shape={out.shape}  nnz={post_pos}  max={post_max}  dtype={out_dtype.__name__}")
    print(f"  affine ref: {ref_path}")


if __name__ == "__main__":
    main()
