"""Produce 4 flip-variant nii.gz outputs of the source Abeta channel through
manual IAR->RPI conversion. User opens each alongside the level-3 SPIM nifti
in their viewer and reports which one overlays correctly.

All four start with transpose (z, y, x) -> (x, y, z), then apply:
  v_noflip : no axis flips
  v_flipy  : flip axis 1 (y)   <- what I tried before
  v_flipx  : flip axis 0 (x)
  v_flipz  : flip axis 2 (z)

Saved at the coarse resolution (12x12x8 um) into the same reference's RPI
affine structure, then resampled onto the reference grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nibabel as nib
import nibabel.processing
import dask.array as da
from zarrnii import ZarrNii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-zarr", required=True, type=Path)
    ap.add_argument("--reference-nii", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = nib.load(str(args.reference_nii))
    print(f"reference shape: {ref.shape}  axcodes: {nib.aff2axcodes(ref.affine)}")

    # Load source, take channel 0, mean-pool by (2, 3, 3)
    zm = ZarrNii.from_ome_zarr(str(args.source_zarr))
    print(f"source shape: {zm.darr.shape}  zarrnii xyz_orientation: {zm.xyz_orientation}")
    src = zm.darr[0:1]      # (1, Z, Y, X)
    _, Z, Y, X = src.shape
    fz, fy, fx = 2, 3, 3
    pad_z = (fz - Z % fz) % fz
    pad_y = (fy - Y % fy) % fy
    pad_x = (fx - X % fx) % fx
    src_p = da.pad(src, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x))) if (pad_z or pad_y or pad_x) else src
    coarse = da.coarsen(da.mean, src_p, {0: 1, 1: fz, 2: fy, 3: fx}).compute().astype(np.float32)
    print(f"coarse shape: {coarse.shape}  range [{coarse.min():.0f}, {coarse.max():.0f}]")

    # Strip channel: (Zc, Yc, Xc)
    data_zyx = coarse[0]

    # Common: transpose to (Xc, Yc, Zc)
    base = data_zyx.transpose(2, 1, 0)
    print(f"after transpose shape: {base.shape}")

    # Affine in level-3 RPI convention (matches reference structurally):
    vx_mm = 0.004 * fx  # 0.012
    vy_mm = 0.004 * fy  # 0.012
    vz_mm = 0.004 * fz  # 0.008
    aff = np.array([
        [ vx_mm, 0.0,    0.0,    0.0],
        [ 0.0,  -vy_mm,  0.0,    0.0],
        [ 0.0,   0.0,   -vz_mm,  0.0],
        [ 0.0,   0.0,    0.0,    1.0],
    ], dtype=float)
    print(f"coarse affine:\n{aff}")

    variants = {
        "noflip": base,
        "flipx":  np.flip(base, axis=0),
        "flipy":  np.flip(base, axis=1),
        "flipz":  np.flip(base, axis=2),
        "flipxy": np.flip(np.flip(base, axis=0), axis=1),
        "flipxz": np.flip(np.flip(base, axis=0), axis=2),
        "flipyz": np.flip(np.flip(base, axis=1), axis=2),
        "flipxyz": np.flip(np.flip(np.flip(base, axis=0), axis=1), axis=2),
    }

    for tag, arr in variants.items():
        arr = np.ascontiguousarray(arr).astype(np.float32)
        coarse_nii = nib.Nifti1Image(arr, aff)
        out_nii = nibabel.processing.resample_from_to(coarse_nii, ref, order=0)
        out_data = out_nii.get_fdata().astype(np.float32)
        out_path = args.out_dir / f"VARIANT_{tag}_source_intensity.nii.gz"
        nib.save(nib.Nifti1Image(out_data, out_nii.affine), str(out_path))
        nz_in_brain = float(((ref.get_fdata() > 50) & (out_data > 50)).sum())
        nz_total = float((out_data > 50).sum())
        ref_nz = float((ref.get_fdata() > 50).sum())
        overlap_pct = 100 * nz_in_brain / max(1, ref_nz)
        print(f"  {tag}:  saved  mean={out_data.mean():.1f}  "
              f"foreground voxels (>50)={int(nz_total)}  "
              f"overlap with SPIM brain: {int(nz_in_brain)} / {int(ref_nz)} = {overlap_pct:.1f}%")


if __name__ == "__main__":
    main()
