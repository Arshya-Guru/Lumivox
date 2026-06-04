"""Tile a CANVAS subject into 128^3 patches at 4um isotropic, and produce a
matching full-res patch dict covering the same physical extent per patch.

Two manifests are written:
    manifests/canvas_<subject_lc>_4um.json     # 128^3 entries at the resampled-4um zarr
    manifests/canvas_<subject_lc>_fullres.json # matching entries in the full-res native zarr

Plus a few side-by-side QC PNGs comparing the same patch in both resolutions:
    qc_images/canvas_<subject_lc>_pairs/<patch_id>.png

Usage:
    pixi run python scripts/build_canvas_patches.py
    pixi run python scripts/build_canvas_patches.py --subject sub-IBA1brain11 --qc 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PATCH = 128  # 4um patch side in voxels


# ---------------------------------------------------------------------------
# Metadata helpers (no array reads)
# ---------------------------------------------------------------------------

def read_fullres_meta(zarr_root: Path) -> Tuple[List[int], List[float]]:
    """OME-Zarr v0.5 (zarr v3): zarr.json at root + s0/zarr.json with shape."""
    grp = json.loads((zarr_root / "zarr.json").read_text())
    ms = grp["attributes"]["ome"]["multiscales"][0]
    s0_scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    s0_shape = json.loads((zarr_root / "s0" / "zarr.json").read_text())["shape"]
    return s0_shape, s0_scale  # [c,z,y,x]


def read_resampled_meta(zarr_root: Path) -> Tuple[List[int], List[float]]:
    """OME-Zarr v0.4 (zarr v2): .zattrs at root + 0/.zarray with shape."""
    grp = json.loads((zarr_root / ".zattrs").read_text())
    ms = grp["multiscales"][0]
    s0_scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    s0_shape = json.loads((zarr_root / "0" / ".zarray").read_text())["shape"]
    return s0_shape, s0_scale  # [c,z,y,x]


def patch_starts(extent: int, patch: int) -> List[int]:
    """Non-overlapping starts; drop the trailing sliver if < patch.
    (User asked to "split the whole image"; this is the simplest interpretation —
    full 128-voxel patches only, no edge clamping. Most edge patches would be
    background-only anyway.)
    """
    n = max(1, (extent - patch) // patch + 1)
    return [i * patch for i in range(n)]


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def build_manifests(
    subject: str,
    fullres_zarr: Path,
    res4_zarr: Path,
    out_dir: Path,
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    fr_shape, fr_scale = read_fullres_meta(fullres_zarr)
    r4_shape, r4_scale = read_resampled_meta(res4_zarr)

    # Sanity: physical extents must match
    for i, axis in zip((1, 2, 3), ("z", "y", "x")):
        fr_ext = fr_shape[i] * fr_scale[i]
        r4_ext = r4_shape[i] * r4_scale[i]
        if abs(fr_ext - r4_ext) > 0.01:  # > 10 um disagreement is a real problem
            raise RuntimeError(
                f"physical extent mismatch on axis {axis}: "
                f"fullres={fr_ext:.4f} mm  vs  resampled={r4_ext:.4f} mm"
            )

    # Voxel-ratio (resampled vox * ratio = fullres vox)
    ratio = [r4_scale[i] / fr_scale[i] for i in range(4)]  # c,z,y,x

    # 4um grid
    _, z4, y4, x4 = r4_shape
    z_starts = patch_starts(z4, PATCH)
    y_starts = patch_starts(y4, PATCH)
    x_starts = patch_starts(x4, PATCH)

    # Fullres patch shape (matching same physical box)
    fr_patch_shape = [
        int(round(PATCH * ratio[1])),
        int(round(PATCH * ratio[2])),
        int(round(PATCH * ratio[3])),
    ]

    entries_4um: List[Dict[str, Any]] = []
    entries_fr: List[Dict[str, Any]] = []
    for gz, zs in enumerate(z_starts):
        for gy, ys in enumerate(y_starts):
            for gx, xs in enumerate(x_starts):
                ze, ye, xe = zs + PATCH, ys + PATCH, xs + PATCH
                patch_id = f"patch_z{gz:02d}_y{gy:02d}_x{gx:02d}"

                # Physical bbox in mm at this patch
                phys_origin = [zs * r4_scale[1], ys * r4_scale[2], xs * r4_scale[3]]
                phys_size = [PATCH * r4_scale[1], PATCH * r4_scale[2], PATCH * r4_scale[3]]

                # Matching fullres voxel box (same physical extent)
                fz_s = int(round(zs * ratio[1]))
                fy_s = int(round(ys * ratio[2]))
                fx_s = int(round(xs * ratio[3]))
                fz_e = fz_s + fr_patch_shape[0]
                fy_e = fy_s + fr_patch_shape[1]
                fx_e = fx_s + fr_patch_shape[2]

                entries_4um.append({
                    "patch_id": patch_id,
                    "grid_index": [gz, gy, gx],
                    "vox_start": [zs, ys, xs],
                    "vox_end": [ze, ye, xe],
                    "vox_shape": [PATCH, PATCH, PATCH],
                    "phys_origin_mm": phys_origin,
                    "phys_size_mm": phys_size,
                })
                entries_fr.append({
                    "patch_id": patch_id,
                    "grid_index": [gz, gy, gx],
                    "vox_start": [fz_s, fy_s, fx_s],
                    "vox_end": [fz_e, fy_e, fx_e],
                    "vox_shape": fr_patch_shape,
                    "phys_origin_mm": phys_origin,
                    "phys_size_mm": phys_size,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    subj_lc = subject.replace("sub-", "").lower()

    def write(name: str, entries: List[Dict[str, Any]], shape, scale, zarr_path: Path):
        path = out_dir / f"canvas_{subj_lc}_{name}.json"
        path.write_text(json.dumps({
            "config": {
                "subject": subject,
                "zarr_path": str(zarr_path),
                "resolution_level": "s0",
                "zarr_shape_czyx": shape,
                "voxel_scale_mm_czyx": scale,
                "patch_shape_vox_zyx": [PATCH, PATCH, PATCH] if name == "4um"
                                       else fr_patch_shape,
                "grid_shape": [len(z_starts), len(y_starts), len(x_starts)],
                "stride": "non-overlapping; trailing edge slivers dropped",
            },
            "patches": entries,
        }, indent=2))
        return path

    p_4um = write("4um", entries_4um, r4_shape, r4_scale, res4_zarr)
    p_fr = write("fullres", entries_fr, fr_shape, fr_scale, fullres_zarr)

    print(f"Tiling grid: {len(z_starts)} z × {len(y_starts)} y × {len(x_starts)} x "
          f"= {len(entries_4um):,} patches")
    print(f"  4um patch shape (zyx):     {PATCH} × {PATCH} × {PATCH}")
    print(f"  fullres patch shape (zyx): {fr_patch_shape[0]} × {fr_patch_shape[1]} × {fr_patch_shape[2]}")
    print(f"  wrote: {p_4um}")
    print(f"  wrote: {p_fr}")
    return p_4um, p_fr, entries_4um


# ---------------------------------------------------------------------------
# QC: read a few patches from each zarr and save side-by-side PNGs
# ---------------------------------------------------------------------------

def render_qc(
    subject: str,
    fullres_zarr: Path,
    res4_zarr: Path,
    qc_dir: Path,
    n_qc: int,
    seed: int = 0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zarr

    subj_lc = subject.replace("sub-", "").lower()
    ents_4um = json.loads(Path(f"manifests/canvas_{subj_lc}_4um.json").read_text())["patches"]
    ents_fr  = json.loads(Path(f"manifests/canvas_{subj_lc}_fullres.json").read_text())["patches"]
    by_id_fr = {e["patch_id"]: e for e in ents_fr}

    # zarr v3 (fullres) and v2 (resampled) — zarr.open handles both
    fr_arr = zarr.open(str(fullres_zarr / "s0"), mode="r")
    r4_arr = zarr.open(str(res4_zarr / "0"), mode="r")

    # Pick patches biased toward the centre of the volume so we don't waste time
    # on background-only ones for QC. Take the middle slab in z, random in y/x.
    zs = sorted({e["grid_index"][0] for e in ents_4um})
    mid_z = zs[len(zs) // 2]
    center_patches = [e for e in ents_4um if e["grid_index"][0] == mid_z]
    rng = random.Random(seed)
    rng.shuffle(center_patches)

    qc_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for entry in center_patches:
        pid = entry["patch_id"]
        z0, y0, x0 = entry["vox_start"]
        z1, y1, x1 = entry["vox_end"]
        r4 = np.asarray(r4_arr[0, z0:z1, y0:y1, x0:x1])
        if r4.size == 0:
            continue
        # Skip mostly-empty patches for QC
        if float(r4.max()) < 50:
            continue

        fe = by_id_fr[pid]
        fz0, fy0, fx0 = fe["vox_start"]
        fz1, fy1, fx1 = fe["vox_end"]
        fr = np.asarray(fr_arr[0, fz0:fz1, fy0:fy1, fx0:fx1])

        mid = r4.shape[0] // 2
        fmid = fr.shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        vmax_r4 = float(np.percentile(r4, 99.5)) or 1
        vmax_fr = float(np.percentile(fr, 99.5)) or 1
        axes[0, 0].imshow(r4[mid], cmap="gray", vmin=0, vmax=vmax_r4)
        axes[0, 0].set_title(f"4um  axial (z={mid}/{r4.shape[0]})  shape={r4.shape}")
        axes[0, 1].imshow(r4[:, r4.shape[1] // 2, :], cmap="gray", vmin=0, vmax=vmax_r4)
        axes[0, 1].set_title("4um  coronal")
        axes[0, 2].imshow(r4[:, :, r4.shape[2] // 2], cmap="gray", vmin=0, vmax=vmax_r4)
        axes[0, 2].set_title("4um  sagittal")

        axes[1, 0].imshow(fr[fmid], cmap="gray", vmin=0, vmax=vmax_fr)
        axes[1, 0].set_title(f"fullres axial (z={fmid}/{fr.shape[0]})  shape={fr.shape}")
        axes[1, 1].imshow(fr[:, fr.shape[1] // 2, :], cmap="gray", vmin=0, vmax=vmax_fr)
        axes[1, 1].set_title("fullres coronal")
        axes[1, 2].imshow(fr[:, :, fr.shape[2] // 2], cmap="gray", vmin=0, vmax=vmax_fr)
        axes[1, 2].set_title("fullres sagittal")

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{subject}  {pid}  (same physical extent: "
                     f"{entry['phys_size_mm'][0]*1000:.0f}×"
                     f"{entry['phys_size_mm'][1]*1000:.0f}×"
                     f"{entry['phys_size_mm'][2]*1000:.0f} um)")
        out = qc_dir / f"{pid}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  qc: {out}")

        rendered += 1
        if rendered >= n_qc:
            break

    if rendered == 0:
        print("  qc: no informative patches found (all centre-slab patches were empty)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="sub-IBA1brain11")
    p.add_argument("--canvas-root", default="/nfs/khan/datasets/CANVAS/bids")
    p.add_argument("--manifests-dir", default="manifests")
    p.add_argument("--qc-dir", default="qc_images")
    p.add_argument("--qc", type=int, default=3, help="number of QC PNGs to render (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bids = Path(args.canvas_root)
    micr_fr = bids / args.subject / "micr"
    micr_r4 = bids / "derivatives" / "resampled" / args.subject / "micr"
    fullres_zarr = next(micr_fr.glob(f"{args.subject}_*SPIM.ome.zarr"))
    res4_zarr = next(micr_r4.glob(f"{args.subject}_*res-4um*SPIM.ome.zarr"))

    print(f"subject: {args.subject}")
    print(f"  fullres:  {fullres_zarr}")
    print(f"  4um:      {res4_zarr}")

    build_manifests(args.subject, fullres_zarr, res4_zarr, Path(args.manifests_dir))

    if args.qc > 0:
        qc_dir = Path(args.qc_dir) / f"canvas_{args.subject.replace('sub-','').lower()}_pairs"
        render_qc(args.subject, fullres_zarr, res4_zarr, qc_dir, args.qc, seed=args.seed)


if __name__ == "__main__":
    main()
