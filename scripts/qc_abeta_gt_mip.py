"""Render an axial-MIP QC PNG for every Abeta gold patch at $GT/Abeta.

Each PNG has two panels (left = raw MIP; right = raw MIP + gold mask overlay
in red @ 0.5 alpha). Files are written to $GT/Abeta/qc/{patch_id}.png.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def render_one(crop_path: Path, gold_path: Path, out_path: Path):
    raw = nib.load(str(crop_path)).get_fdata()
    gold = nib.load(str(gold_path)).get_fdata()

    # Drop singleton dims (some niftis come back with a length-1 channel axis)
    raw = np.squeeze(raw)
    gold = np.squeeze(gold)
    if raw.ndim != 3:
        raise RuntimeError(f"unexpected raw ndim {raw.ndim} for {crop_path}")

    # Axial MIP — project along the first axis of the array (this is z in the
    # zarr/nifti convention used everywhere else in this repo)
    raw_mip = raw.max(axis=0)
    gold_mip = (gold > 0).any(axis=0).astype(np.uint8)

    vmax = float(np.percentile(raw_mip, 99.5)) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    axes[0].imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title(f"raw axial MIP  shape={raw.shape}  max={raw.max():.0f}")

    axes[1].imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax)
    # Overlay gold as red translucent layer (masked array so zeros stay transparent)
    overlay = np.ma.masked_where(gold_mip == 0, gold_mip)
    axes[1].imshow(overlay, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    pos = int(gold_mip.sum())
    total_3d = int((gold > 0).sum())
    axes[1].set_title(
        f"raw MIP + gold overlay  "
        f"(mask voxels 3D={total_3d}  2D-MIP={pos})"
    )
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gt-root",
        default=os.environ.get("GT", "/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized"),
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    abeta_root = Path(args.gt_root) / "Abeta"
    out_dir = abeta_root / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect (crop, gold) pairs across all datasets/subjects
    pairs = []
    for gold in sorted(abeta_root.rglob("*_seg_gold.nii.gz")):
        # patch_id = filename without _seg_gold.nii.gz suffix
        pid = gold.name.replace("_seg_gold.nii.gz", "")
        crop = gold.parent / f"{pid}_crop128.nii.gz"
        if crop.exists():
            pairs.append((pid, crop, gold))
        # AS37F4 oddity: gold filename uses 'AS' prefix but crop uses 'A'
        elif gold.name.startswith("sub-AS37F4_") and (gold.parent / "sub-A37F4_patch00_crop128.nii.gz").exists():
            crop_alt = gold.parent / "sub-A37F4_patch00_crop128.nii.gz"
            pairs.append((pid, crop_alt, gold))
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Found {len(pairs)} (crop, gold) pairs in {abeta_root}")
    for i, (pid, crop, gold) in enumerate(pairs, 1):
        out = out_dir / f"{pid}.png"
        try:
            render_one(crop, gold, out)
            print(f"  [{i}/{len(pairs)}] {pid}: {out}")
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] {pid}: FAILED ({exc})", file=sys.stderr)

    print(f"\nWrote {len(pairs)} MIP PNGs to {out_dir}")


if __name__ == "__main__":
    main()
