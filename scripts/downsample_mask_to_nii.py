"""Downsample an OME-Zarr binary mask to match a reference NIfTI's grid, in
the reference's physical/anatomical frame, and write as a NIfTI file.

Pipeline:
  1. Load mask via zarrnii.
  2. Max-pool downsample by integer factors (preserves sparse binary fg).
  3. Manually convert IAR (z, y, x) -> RPI (x, y-flipped, z) — this matches
     the physical-frame convention of the level-3 SPIM nifti.
  4. Build a diagonal RPI affine using the level-3 voxel-size convention.
  5. Resample onto the reference grid via nibabel (order=0 nearest).
  6. Save as nii.gz.

We don't use zarrnii.to_nifti() here because zarrnii reads xyz_orientation
in (X, Y, Z) order, but the source ome.zarr's "IAR" tag is intended in data-
dimension order (z, y, x). The level-3 SPIM nifti follows the latter
interpretation (matching physical extents), so we do the conversion manually
to match it.

Usage:
    python scripts/downsample_mask_to_nii.py \\
        --mask-zarr path/to/mask.ome.zarr \\
        --reference-nii path/to/level-3.nii \\
        --output-nii path/to/output.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `from lumivox.X import Y` work when this script is run by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nibabel as nib
import nibabel.processing
import dask.array as da

from zarrnii import ZarrNii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-zarr", required=True, type=Path)
    ap.add_argument("--reference-nii", required=True, type=Path)
    ap.add_argument("--output-nii", required=True, type=Path)
    ap.add_argument("--reduction", default="max", choices=["max", "mean"],
                    help="Block reduction (max for binary masks; mean for intensity).")
    args = ap.parse_args()

    print(f"Mask/source: {args.mask_zarr}")
    print(f"Reference:   {args.reference_nii}")
    print(f"Output:      {args.output_nii}")
    print(f"Reduction:   {args.reduction}")

    # ----- Reference grid -----
    ref = nib.load(str(args.reference_nii))
    print(f"Reference shape: {ref.shape}  voxels (mm): {ref.header.get_zooms()}")
    print(f"Reference affine:\n{ref.affine}")
    print(f"Reference axcodes: {nib.aff2axcodes(ref.affine)}")

    # ----- Load mask via ZarrNii (just for the dask array; we do orientation ourselves) -----
    print("\nLoading source via ZarrNii...")
    zm = ZarrNii.from_ome_zarr(str(args.mask_zarr))
    print(f"  source shape: {zm.darr.shape}  zarrnii xyz_orientation: {zm.xyz_orientation}")
    src_darr = zm.darr   # (C, Z, Y, X) dask uint8
    C, Z, Y, X = src_darr.shape

    # ----- Max- or mean-pool to coarse resolution -----
    fz, fy, fx = 2, 3, 3     # source 4 um  -> coarse (8, 12, 12) um
    print(f"\nPre-downsampling factors (z, y, x) via {args.reduction}-pool: ({fz}, {fy}, {fx})")
    pad_z = (fz - Z % fz) % fz
    pad_y = (fy - Y % fy) % fy
    pad_x = (fx - X % fx) % fx
    if pad_z or pad_y or pad_x:
        src_p = da.pad(src_darr, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
                       mode="constant", constant_values=0)
    else:
        src_p = src_darr
    op = da.max if args.reduction == "max" else da.mean
    coarse = da.coarsen(op, src_p, {0: 1, 1: fz, 2: fy, 3: fx})
    print(f"  coarse shape: {coarse.shape}")
    coarse_np = coarse.compute().astype(np.uint8 if args.reduction == "max" else np.float32)
    if args.reduction == "max":
        nz = int((coarse_np > 0).sum())
        print(f"  coarse nonzero (max-pool): {nz}")

    # ----- Manual conversion: just transpose, no flip -----
    # The source ome.zarr's xyz_orientation tag says "IAR" but empirically the
    # data is in RPI orientation already (verified by 99% overlap of source
    # intensity with the level-3 SPIM nifti under noflip; all flip combinations
    # gave worse overlap). So:
    #   source z (axis 0) = I direction  (z=0 Superior, z=max Inferior)
    #   source y (axis 1) = P direction  (y=0 Anterior, y=max Posterior)
    #   source x (axis 2) = R direction  (x=0 Left, x=max Right)
    print("\nManual conversion (transpose ZYX->XYZ, no flip)...")
    data_zyx = coarse_np[0]                                       # (Zc, Yc, Xc)
    data_rpi = np.ascontiguousarray(data_zyx.transpose(2, 1, 0))  # (Xc, Yc, Zc) = (R, P, I)
    print(f"  data_rpi shape: {data_rpi.shape}")

    # ----- Coarse affine in RAI convention (matches level-5 SPIM nifti, OG truth) -----
    # OG's xyz_orientation: "RAI" gives axes (R, A, I) after transpose ZYX->XYZ.
    # Level-5 SPIM nifti has the same: aff2axcodes ('R', 'A', 'I').
    # Level-3 SPIM nifti is A-P mirrored (P instead of A) — broken pipeline.
    vx_mm = 0.004 * fx   # 12 um -> 0.012 mm in i (R direction)
    vy_mm = 0.004 * fy   # 12 um -> 0.012 mm in j (A direction)
    vz_mm = 0.004 * fz   #  8 um -> 0.008 mm in k (I direction)
    aff_rai = np.array([
        [ vx_mm, 0.0,    0.0,   0.0],   # i -> +X (R)
        [ 0.0,  +vy_mm,  0.0,   0.0],   # j -> +Y (A)
        [ 0.0,   0.0,   -vz_mm, 0.0],   # k -> -Z (I)
        [ 0.0,   0.0,    0.0,   1.0],
    ], dtype=float)
    print(f"  coarse RAI affine:\n{aff_rai}")

    coarse_nii = nib.Nifti1Image(data_rpi, aff_rai)
    print(f"  coarse nii axcodes: {nib.aff2axcodes(coarse_nii.affine)}")

    # ----- Resample onto reference grid -----
    print("\nResampling onto reference grid (order=0 nearest)...")
    out_nii = nibabel.processing.resample_from_to(coarse_nii, ref, order=0)
    out_data = out_nii.get_fdata()
    out_data = out_data.astype(np.uint8 if args.reduction == "max" else np.float32)
    print(f"  final shape: {out_data.shape}")
    if args.reduction == "max":
        print(f"  final nonzero: {int((out_data > 0).sum())}")
    else:
        print(f"  final range: [{out_data.min():.0f}, {out_data.max():.0f}]  mean: {out_data.mean():.1f}")

    args.output_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_data, out_nii.affine), str(args.output_nii))
    print(f"  saved: {args.output_nii}  ({args.output_nii.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
