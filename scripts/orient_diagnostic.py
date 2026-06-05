"""Diagnostic: save BOTH the source Abeta channel and the mask as niftis
through the same zarrnii pipeline, then check whether the source-channel
nifti aligns with the level-3 SPIM nifti.

If it does -> my mask approach is correct (zarrnii convention matches level-3).
If not    -> there's a convention mismatch (zarrnii vs level-3 maker) and we
             need to use level-3's affine directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `from lumivox.X import Y` work when this script is run by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nibabel as nib
import dask.array as da
from zarrnii import ZarrNii


def save_via_zarrnii(zarr_path: str, channel: int, fz: int, fy: int, fx: int,
                     reduction: str, out_nii: str):
    """Load a 4D OME-Zarr, max- or mean-pool by (fz, fy, fx), save via zarrnii.to_nifti.

    Operates on ALL channels through zarrnii's downsample (to keep the
    affine correct) and selects `channel` from the final 3D/4D nifti.
    """
    zm = ZarrNii.from_ome_zarr(zarr_path)
    print(f"  source shape: {zm.darr.shape}  channel taken (final): {channel}")
    src = zm.darr           # (C, Z, Y, X)
    C, Z, Y, X = src.shape

    # zarrnii downsample preserves all channels — get the correct target shape
    zm_ds = zm.downsample(factors=[fz, fy, fx])
    target = zm_ds.darr.shape   # (C, Zc, Yc, Xc)

    # Pad source so coarsen matches zarrnii's ceil-shape
    pad_z = (fz - Z % fz) % fz
    pad_y = (fy - Y % fy) % fy
    pad_x = (fx - X % fx) % fx
    src_p = da.pad(src, ((0,0), (0,pad_z), (0,pad_y), (0,pad_x))) if (pad_z or pad_y or pad_x) else src
    op = da.max if reduction == "max" else da.mean
    coarse = da.coarsen(op, src_p, {0: 1, 1: fz, 2: fy, 3: fx})
    assert coarse.shape == target, f"shape mismatch: {coarse.shape} vs {target}"
    zm_ds.darr = coarse

    # Materialize via zarrnii.to_nifti
    out_nii_obj_all = zm_ds.to_nifti()
    print(f"  zarrnii nifti shape (all chans): {out_nii_obj_all.shape}  affine:\n{out_nii_obj_all.affine}")

    # zarrnii.to_nifti for multi-channel data returns shape (X, Y, Z, C). Select channel.
    data = out_nii_obj_all.get_fdata()
    if data.ndim == 4 and C > 1:
        data = data[..., channel]
        print(f"  selected channel {channel} -> shape {data.shape}")

    out_nii_obj = nib.Nifti1Image(data, out_nii_obj_all.affine)
    out_path = Path(out_nii)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii_obj, str(out_path))
    print(f"  saved: {out_path}  shape={out_nii_obj.shape}")
    return out_path, out_nii_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-zarr", required=True)
    ap.add_argument("--mask-zarr", required=True)
    ap.add_argument("--reference-nii", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same coarse factors for both
    fz, fy, fx = 2, 3, 3

    print("=== SAVING SOURCE ABETA CHANNEL (channel 0) ===")
    src_nii_path, src_nii = save_via_zarrnii(
        args.source_zarr, channel=0, fz=fz, fy=fy, fx=fx, reduction="mean",
        out_nii=str(args.out_dir / "DIAG_source_abeta_via_zarrnii.nii.gz"),
    )

    print("\n=== SAVING MASK (channel 0) ===")
    mask_nii_path, mask_nii = save_via_zarrnii(
        args.mask_zarr, channel=0, fz=fz, fy=fy, fx=fx, reduction="max",
        out_nii=str(args.out_dir / "DIAG_mask_via_zarrnii.nii.gz"),
    )

    print("\n=== REFERENCE LEVEL-3 SPIM NIFTI ===")
    ref = nib.load(args.reference_nii)
    print(f"  ref shape: {ref.shape}  affine:\n{ref.affine}")

    print("\n=== ALIGNMENT CHECK ===")
    print(f"src nifti affine axcodes: {nib.aff2axcodes(src_nii.affine)}")
    print(f"mask nifti affine axcodes: {nib.aff2axcodes(mask_nii.affine)}")
    print(f"ref nifti affine axcodes: {nib.aff2axcodes(ref.affine)}")

    # Quick check: do the two have meaningful overlap when resampled to ref?
    print("\nResampling source-abeta-channel onto ref grid...")
    import nibabel.processing
    resampled_src = nibabel.processing.resample_from_to(src_nii, ref, order=0)
    resampled_src_data = resampled_src.get_fdata()
    print(f"  resampled src shape: {resampled_src_data.shape}")
    print(f"  resampled src range: [{resampled_src_data.min():.0f}, {resampled_src_data.max():.0f}]")
    print(f"  resampled src mean: {resampled_src_data.mean():.1f}")
    print(f"  reference SPIM mean: {ref.get_fdata().mean():.1f}")

    print("\nResampling mask onto ref grid...")
    resampled_mask = nibabel.processing.resample_from_to(mask_nii, ref, order=0)
    resampled_mask_data = resampled_mask.get_fdata().astype(np.uint8)
    print(f"  resampled mask nonzero: {int((resampled_mask_data > 0).sum())}")

    # Save the resampled niftis for visual inspection
    nib.save(resampled_src, str(args.out_dir / "DIAG_source_abeta_resampled_to_ref.nii.gz"))
    nib.save(nib.Nifti1Image(resampled_mask_data, resampled_mask.affine),
             str(args.out_dir / "DIAG_mask_resampled_to_ref.nii.gz"))
    print(f"\nDiagnostic outputs in {args.out_dir}:")
    for f in sorted(args.out_dir.glob("DIAG_*.nii.gz")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
