"""Generate QC images for every patch in a manifest.

For each patch: 5-row figure with atlas zooms (3 ortho), zarr zooms, and patch slices.
Saves one PNG per patch into an output directory.

Usage:
    pixi run python scripts/generate_qc_images.py \
        --manifest manifests/qc_3per_subject.json \
        --output qc_images/crop96 \
        --patch-size 256 --crop-size 96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure lumivox is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from zarrnii import ZarrNii, ZarrNiiAtlas
from lumivox.data.manifest import resolve_stain_channel


def generate_qc_image(
    patch_entry: dict,
    patch_idx: int,
    patch_size: int,
    output_dir: Path,
):
    sub_id = patch_entry["subject_id"]
    zarr_path = patch_entry["zarr_path"]
    stain_ch = patch_entry["stain_channel"]
    center_vox = patch_entry["center_vox"]
    center_phys = patch_entry["center_phys"]
    dseg_path = patch_entry["dseg_path"]
    labels_path = patch_entry["labels_path"]
    dataset_name = Path(patch_entry["dataset_root"]).name

    # Load atlas
    atlas = ZarrNiiAtlas.from_files(dseg_path=dseg_path, labels_path=labels_path)
    dseg_data = atlas.dseg.darr.compute().squeeze()
    target_ids = {3, 4, 5, 6}
    target_mask = np.isin(dseg_data, list(target_ids))
    dseg_masked = np.ma.masked_where(dseg_data == 0, dseg_data)

    # Dseg voxel coords
    inv_affine = atlas.dseg.affine.invert()
    vox_dseg = inv_affine @ np.array(center_phys)
    di, dj, dk = int(round(vox_dseg[0])), int(round(vox_dseg[1])), int(round(vox_dseg[2]))

    label_val = dseg_data[di, dj, dk]
    label_name = atlas.labels_df[atlas.labels_df["index"] == int(label_val)]["name"].values
    label_str = label_name[0] if len(label_name) > 0 else f"label={int(label_val)}"

    # Load zarr
    znimg = ZarrNii.from_ome_zarr(zarr_path, channels=[stain_ch])
    darr = znimg.darr

    # Resampled voxel coords
    cz, cy, cx = int(round(center_vox[0])), int(round(center_vox[1])), int(round(center_vox[2]))

    # Extract patch
    half = patch_size // 2
    z0, z1 = max(0, cz - half), min(darr.shape[1], cz + half)
    y0, y1 = max(0, cy - half), min(darr.shape[2], cy + half)
    x0, x1 = max(0, cx - half), min(darr.shape[3], cx + half)
    patch = darr[0, z0:z1, y0:y1, x0:x1].compute().astype(np.float32)
    patch_vmin, patch_vmax = np.percentile(patch, [1, 99])

    # --- Plot ---
    atlas_pads = [None, 60, 30, 15]
    fig, axes = plt.subplots(5, 4, figsize=(26, 32))

    # Row 0: Atlas dim0
    for col, pad in enumerate(atlas_pads):
        ax = axes[0, col]
        ax.imshow(dseg_masked[di], cmap="nipy_spectral", vmin=1, vmax=22, origin="lower")
        ax.contour(target_mask[di], colors="lime", linewidths=0.8)
        ax.plot(dk, dj, "ro", markersize=10, markeredgecolor="white", markeredgewidth=2)
        if pad is not None:
            ax.set_xlim(dk - pad, dk + pad)
            ax.set_ylim(dj - pad, dj + pad)
        ax.set_title(f"dim0={di}" + (f" ±{pad}" if pad else " full"))

    # Row 1: Atlas dim1
    for col, pad in enumerate(atlas_pads):
        ax = axes[1, col]
        ax.imshow(dseg_masked[:, dj, :], cmap="nipy_spectral", vmin=1, vmax=22, origin="lower")
        ax.contour(target_mask[:, dj, :], colors="lime", linewidths=0.8)
        ax.plot(dk, di, "ro", markersize=10, markeredgecolor="white", markeredgewidth=2)
        if pad is not None:
            ax.set_xlim(dk - pad, dk + pad)
            ax.set_ylim(di - pad, di + pad)
        ax.set_title(f"dim1={dj}" + (f" ±{pad}" if pad else " full"))

    # Row 2: Atlas dim2
    for col, pad in enumerate(atlas_pads):
        ax = axes[2, col]
        ax.imshow(dseg_masked[:, :, dk], cmap="nipy_spectral", vmin=1, vmax=22, origin="lower")
        ax.contour(target_mask[:, :, dk], colors="lime", linewidths=0.8)
        ax.plot(dj, di, "ro", markersize=10, markeredgecolor="white", markeredgewidth=2)
        if pad is not None:
            ax.set_xlim(dj - pad, dj + pad)
            ax.set_ylim(di - pad, di + pad)
        ax.set_title(f"dim2={dk}" + (f" ±{pad}" if pad else " full"))

    # Row 3: Zarr axial zooms
    zooms = [
        ("Full FOV", 0, darr.shape[2], 0, darr.shape[3]),
        ("8x", max(0, cy-1024), min(darr.shape[2], cy+1024), max(0, cx-1024), min(darr.shape[3], cx+1024)),
        ("4x", max(0, cy-512), min(darr.shape[2], cy+512), max(0, cx-512), min(darr.shape[3], cx+512)),
        ("2x", max(0, cy-256), min(darr.shape[2], cy+256), max(0, cx-256), min(darr.shape[3], cx+256)),
    ]

    for col, (label, ry0, ry1, rx0, rx1) in enumerate(zooms):
        slab = darr[0, cz, ry0:ry1, rx0:rx1].compute().astype(np.float32)
        ax = axes[3, col]
        ax.imshow(slab, cmap="gray", vmin=patch_vmin, vmax=patch_vmax, origin="lower",
                  extent=[rx0, rx1, ry0, ry1])
        rect = Rectangle((x0, y0), x1-x0, y1-y0, lw=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"Zarr z={cz} — {label}")

    # Row 4: 1.2x zoom + 3 patch slices
    slab = darr[0, cz, max(0, cy-160):min(darr.shape[2], cy+160),
                max(0, cx-160):min(darr.shape[3], cx+160)].compute().astype(np.float32)
    ax = axes[4, 0]
    ax.imshow(slab, cmap="gray", vmin=patch_vmin, vmax=patch_vmax, origin="lower",
              extent=[max(0, cx-160), min(darr.shape[3], cx+160),
                      max(0, cy-160), min(darr.shape[2], cy+160)])
    rect = Rectangle((x0, y0), x1-x0, y1-y0, lw=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"Zarr z={cz} — 1.2x")

    mid = patch.shape[0] // 2
    for i, offset in enumerate([-20, 0, 20]):
        ax = axes[4, i + 1]
        sl = mid + offset
        if sl < 0 or sl >= patch.shape[0]:
            ax.axis("off")
            continue
        ax.imshow(patch[sl], cmap="gray",
                  vmin=np.percentile(patch[sl], 0.5), vmax=np.percentile(patch[sl], 99.5))
        ax.set_title(f"Patch z={sl}")
        ax.axis("off")

    fig.suptitle(
        f"{dataset_name} / {sub_id} / patch {patch_idx} | vox ({cz},{cy},{cx}) | {label_str}\n"
        f"Rows 0-2: Atlas 3 ortho × 4 zooms | Row 3: Zarr axial zooms | Row 4: 1.2x + Patch",
        fontsize=14,
    )
    plt.tight_layout()

    out_path = output_dir / f"{dataset_name}_{sub_id}_patch{patch_idx:02d}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _worker(args_tuple):
    """Worker function for multiprocessing."""
    entry, local_idx, patch_size, output_dir, task_num, total = args_tuple
    try:
        out = generate_qc_image(entry, local_idx, patch_size, output_dir)
        print(f"  [{task_num}/{total}] {out.name}")
        return True
    except Exception as e:
        print(f"  [{task_num}/{total}] FAILED {entry['subject_id']} patch {local_idx}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate QC images for patch manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="qc_images")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = manifest["patches"]
    print(f"Generating QC images for {len(patches)} patches -> {output_dir} ({args.workers} workers)")

    # Build task list
    from collections import defaultdict
    by_subject = defaultdict(list)
    for i, p in enumerate(patches):
        by_subject[p["subject_id"]].append((i, p))

    tasks = []
    task_num = 0
    for sub_id in sorted(by_subject):
        for local_idx, (global_idx, entry) in enumerate(by_subject[sub_id]):
            task_num += 1
            tasks.append((entry, local_idx, args.patch_size, output_dir, task_num, len(patches)))

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = pool.map(_worker, tasks)
        n_ok = sum(results)
    else:
        n_ok = sum(_worker(t) for t in tasks)

    print(f"\nDone. {n_ok}/{len(patches)} images in {output_dir}")


if __name__ == "__main__":
    main()
