"""Extract a CD31 (vessel) channel crop matching an existing ft_normalized patch.

Reuses the coordinate chain from build_finetune_normalized.py: reads the
patch's meta.json for the center_vox already resolved against the resampled
zarr, picks the CD31 channel from the same zarr, and writes the central 128³
crop next to the existing Abeta crop.

Usage:
    pixi run python scripts/extract_vessel_crop.py \\
        --dataset mouse_app_vaccine_batch \\
        --subject sub-AS176F7 \\
        --patch 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from zarrnii import ZarrNii

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumivox.data.manifest import resolve_stain_channel  # noqa: E402

from build_finetune_normalized import (  # noqa: E402
    CROP_SIZE,
    PATCH_WINDOW,
    central_crop,
    extract_window_from_zarr,
    find_subject_paths,
)


def main(dataset: str, subject: str, patch_num: int, stain: str, out_suffix: str):
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "ft_normalized" / "Abeta" / dataset / subject
    prefix = f"{subject}_patch{patch_num:02d}"
    meta_path = out_dir / f"{prefix}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    center_vox = np.array(meta["center_vox"], dtype=np.float64)
    print(f"  center_vox (from meta) = {center_vox.tolist()}")

    paths = find_subject_paths(dataset, subject)
    if not paths["resampled_zarr"]:
        raise FileNotFoundError(f"No resampled zarr for {dataset}/{subject}")

    stain_ch = resolve_stain_channel(
        paths["resampled_zarr"], stain, paths["fullres_zarr"]
    )
    print(f"  {stain} -> channel {stain_ch} of {paths['resampled_zarr']}")

    znimg = ZarrNii.from_ome_zarr(paths["resampled_zarr"], channels=[stain_ch])
    raw_window, _ = extract_window_from_zarr(znimg, center_vox, size=PATCH_WINDOW)
    raw_crop = central_crop(raw_window, CROP_SIZE)
    print(
        f"  crop shape={raw_crop.shape} "
        f"min={raw_crop.min():.1f} max={raw_crop.max():.1f} mean={raw_crop.mean():.1f} "
        f"nonzero={int((raw_crop > 0).sum())}/{raw_crop.size}"
    )

    affine_4um = np.diag([0.004, 0.004, 0.004, 1.0])
    out_path = out_dir / f"{prefix}_crop128{out_suffix}.nii.gz"
    nib.save(nib.Nifti1Image(raw_crop.astype(np.float32), affine_4um), str(out_path))
    print(f"  wrote {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--patch", type=int, required=True)
    p.add_argument("--stain", default="CD31",
                   help="omero channel label to pull (default: CD31 = vessel)")
    p.add_argument("--suffix", default="_cd31",
                   help="suffix appended to {prefix}_crop128 (default: _cd31)")
    args = p.parse_args()
    main(args.dataset, args.subject, args.patch, args.stain, args.suffix)
