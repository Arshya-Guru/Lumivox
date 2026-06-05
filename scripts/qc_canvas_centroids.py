"""Render full-res axial slices for a few patches that contain CANVAS centroid
annotations, with and without the centroid overlay.

For each chosen patch we save two PNGs in the same canvas_<subj>_pairs/ dir:
    <patch_id>_fullres_axial.png                 -- image only
    <patch_id>_fullres_axial_with_centroids.png  -- same slice + 8um-radius circles

Centroid columns x,y,z in the CSVs are in full-res voxel coords. They get drawn
as alpha=0.5 circles on the axial slice; only centroids within +/- 8um/4um = 2
voxels of the displayed z are drawn (i.e., their 3D ball intersects the slice).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import zarr


RADIUS_UM = 8.0


def load_centroids(csv_dir: Path) -> List[Tuple[int, int, int, str]]:
    """Return list of (x, y, z, region_tag) in full-res voxel coords."""
    centroids = []
    for csv_path in sorted(csv_dir.glob("*_gt.csv")):
        # filename like IBA1_brain11_region1_gt.csv -> tag "region1"
        tag = next((p for p in csv_path.stem.split("_") if p.startswith("region")), csv_path.stem)
        for row in csv.DictReader(open(csv_path)):
            centroids.append((int(row["x"]), int(row["y"]), int(row["z"]), tag))
    return centroids


def patch_contains(patch_box, x, y, z) -> bool:
    """patch_box is fullres (vox_start, vox_end) for z, y, x axes."""
    (z0, y0, x0), (z1, y1, x1) = patch_box
    return (z0 <= z < z1) and (y0 <= y < y1) and (x0 <= x < x1)


def pick_top_patches(manifest_fr, centroids, k_per_region=1, extra=1):
    """For each region tag, pick the patch with the most centroids in it. Returns
    a list of patch entries with the centroid lists attached and the best slice
    z chosen."""
    # Group centroids by region for region-balanced picking
    by_region = {}
    for x, y, z, tag in centroids:
        by_region.setdefault(tag, []).append((x, y, z))

    # For every patch, count centroids per region
    patches = manifest_fr["patches"]
    per_patch_counts = {}  # patch_id -> {region: count}
    per_patch_pts = {}     # patch_id -> [(x,y,z,tag), ...]
    for entry in patches:
        box = (entry["vox_start"], entry["vox_end"])
        hits = []
        for tag, pts in by_region.items():
            for (x, y, z) in pts:
                if patch_contains(box, x, y, z):
                    hits.append((x, y, z, tag))
        if hits:
            per_patch_pts[entry["patch_id"]] = hits
            cnts = {}
            for _, _, _, t in hits:
                cnts[t] = cnts.get(t, 0) + 1
            per_patch_counts[entry["patch_id"]] = cnts

    chosen_ids = []
    # Top from each region
    for tag in sorted(by_region):
        ranked = sorted(
            (pid for pid, cnt in per_patch_counts.items() if tag in cnt),
            key=lambda pid: per_patch_counts[pid][tag],
            reverse=True,
        )
        if ranked:
            chosen_ids.append(ranked[0])
    # Extra patches by total count
    remaining = sorted(
        (pid for pid in per_patch_counts if pid not in chosen_ids),
        key=lambda pid: sum(per_patch_counts[pid].values()),
        reverse=True,
    )
    chosen_ids += remaining[:extra]

    by_id = {e["patch_id"]: e for e in patches}
    out = []
    for pid in chosen_ids:
        entry = by_id[pid]
        out.append((entry, per_patch_pts[pid], per_patch_counts[pid]))
    return out


def best_slice_z(entry, pts, z_radius_vox: int) -> int:
    """Pick the slice (absolute fullres z) with the most centroids in range."""
    z0, y0, x0 = entry["vox_start"]
    z1, _, _ = entry["vox_end"]
    best_z, best_n = (z0 + z1) // 2, -1
    for z in range(z0, z1):
        n = sum(1 for (_, _, zc, _) in pts if abs(zc - z) <= z_radius_vox)
        if n > best_n:
            best_z, best_n = z, n
    return best_z


def render_patch(
    entry,
    pts,
    counts,
    z_slice: int,
    fr_arr,
    out_dir: Path,
    fr_scale_mm,
):
    (z0, y0, x0), (z1, y1, x1) = entry["vox_start"], entry["vox_end"]
    pid = entry["patch_id"]

    # Voxel sizes in mm
    z_um = fr_scale_mm[1] * 1000.0
    y_um = fr_scale_mm[2] * 1000.0
    x_um = fr_scale_mm[3] * 1000.0
    z_radius_vox = int(round(RADIUS_UM / z_um))
    y_radius_vox = RADIUS_UM / y_um
    x_radius_vox = RADIUS_UM / x_um

    # Pull the axial slice (c=0)
    slc = np.asarray(fr_arr[0, z_slice, y0:y1, x0:x1])
    vmax = float(np.percentile(slc, 99.5)) or 1.0

    title_prefix = (
        f"sub-IBA1brain11  {pid}  axial z={z_slice}  "
        f"(centroids in patch: {sum(counts.values())} = " +
        ", ".join(f"{t}:{counts[t]}" for t in sorted(counts)) + ")"
    )

    # PNG 1 -- image only
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(slc, cmap="gray", vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title_prefix)
    out1 = out_dir / f"{pid}_fullres_axial.png"
    fig.savefig(out1, dpi=110)
    plt.close(fig)

    # PNG 2 -- image + centroid overlay (8 um spheres -> only those whose ball
    # intersects this z get drawn)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(slc, cmap="gray", vmin=0, vmax=vmax)
    drawn_per_region = {}
    for (xc, yc, zc, tag) in pts:
        if abs(zc - z_slice) > z_radius_vox:
            continue
        # Centroid in slice coords:
        cx = xc - x0
        cy = yc - y0
        # If the slice is offset in z from the centroid, shrink the in-plane
        # radius accordingly (sphere cross-section): r(slice) = sqrt(R^2 - dz^2)
        dz_um = (zc - z_slice) * z_um
        r_um_plane = (RADIUS_UM ** 2 - dz_um ** 2) ** 0.5
        r_y = r_um_plane / y_um
        r_x = r_um_plane / x_um
        # Use mean of the two as the visual radius (anisotropic voxels but the
        # image is rendered without aspect correction by default so use voxel units)
        ax.add_patch(Circle(
            (cx, cy),
            radius=(r_x + r_y) / 2.0,
            facecolor="red",
            edgecolor="none",
            alpha=0.5,
        ))
        drawn_per_region[tag] = drawn_per_region.get(tag, 0) + 1
    drawn = sum(drawn_per_region.values())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        title_prefix + f"\noverlay: {drawn} centroids drawn (within ±{RADIUS_UM:.0f} um of z)"
    )
    out2 = out_dir / f"{pid}_fullres_axial_with_centroids.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)

    print(f"  {pid}: z={z_slice}, drew {drawn} centroids "
          f"({', '.join(f'{t}:{n}' for t,n in drawn_per_region.items()) or 'none'})")
    return out1, out2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="sub-IBA1brain11")
    p.add_argument(
        "--csv-dir",
        default="/nfs/khan/datasets/CANVAS/derivatives/csv_annotations/sub-IBA1brain11",
    )
    p.add_argument("--manifest-fullres", default=None,
                   help="defaults to manifests/canvas_<subj_lc>_fullres.json")
    p.add_argument("--out-dir", default=None,
                   help="defaults to qc_images/canvas_<subj_lc>_pairs/")
    p.add_argument("--n-patches", type=int, default=4,
                   help="total patches to render (3 from regions + extras)")
    args = p.parse_args()

    subj_lc = args.subject.replace("sub-", "").lower()
    manifest_path = Path(args.manifest_fullres
                         or f"manifests/canvas_{subj_lc}_fullres.json")
    out_dir = Path(args.out_dir or f"qc_images/canvas_{subj_lc}_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_fr = json.loads(manifest_path.read_text())
    fr_scale = manifest_fr["config"]["voxel_scale_mm_czyx"]
    fr_zarr = Path(manifest_fr["config"]["zarr_path"])
    fr_arr = zarr.open(str(fr_zarr / "s0"), mode="r")

    centroids = load_centroids(Path(args.csv_dir))
    n_regions = len({t for _, _, _, t in centroids})
    print(f"Loaded {len(centroids):,} centroids across {n_regions} regions")

    chosen = pick_top_patches(
        manifest_fr,
        centroids,
        k_per_region=1,
        extra=max(0, args.n_patches - n_regions),
    )
    print(f"Top patches by centroid density:")
    for entry, pts, counts in chosen:
        print(f"  {entry['patch_id']}  grid={entry['grid_index']}  "
              f"box(z,y,x)={entry['vox_start']}->{entry['vox_end']}  "
              f"centroids={counts}")

    z_radius_vox = int(round(RADIUS_UM / (fr_scale[1] * 1000.0)))
    for entry, pts, counts in chosen:
        z_slice = best_slice_z(entry, pts, z_radius_vox)
        render_patch(entry, pts, counts, z_slice, fr_arr, out_dir, fr_scale)


if __name__ == "__main__":
    main()
