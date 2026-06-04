"""For each annotated region in CANVAS sub-IBA1brain11, extract one 4um-isotropic
128^3 patch positioned ENTIRELY inside the region, plus the matching full-res
patch (same physical extent, anisotropic voxels) and a full-res sphere mask
of every centroid that falls inside the patch.

Output (under $GT, default
/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized):

    $GT/Iba1/sub-IBA1brain11/region{N}/
        crop_4um.nii.gz                   # 128^3, 4um isotropic
        crop_fullres.nii.gz               # 128 x 284 x 284, z=4um  y=x=1.8um
        centroids_spheres_fullres.nii.gz  # same shape as crop_fullres; binary
                                          #   uint8 with 8 um (physical) spheres
                                          #   rasterised at each centroid
        meta.json                         # patch boxes, voxel sizes, centroid counts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import zarr


PATCH_VOX_4UM = 128
SPHERE_RADIUS_UM = 8.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_centroids(csv_dir: Path) -> Dict[str, List[Tuple[int, int, int]]]:
    """Per-region dict of (x, y, z) tuples in full-res voxel coords."""
    out: Dict[str, List[Tuple[int, int, int]]] = {}
    for f in sorted(csv_dir.glob("*_gt.csv")):
        tag = next(p for p in f.stem.split("_") if p.startswith("region"))
        pts = [(int(r["x"]), int(r["y"]), int(r["z"]))
               for r in csv.DictReader(open(f))]
        out[tag] = pts
    return out


def read_zarr_meta(path: Path, version: str) -> Tuple[List[int], List[float]]:
    if version == "v0.5":
        grp = json.loads((path / "zarr.json").read_text())
        ms = grp["attributes"]["ome"]["multiscales"][0]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        shape = json.loads((path / "s0" / "zarr.json").read_text())["shape"]
    else:
        grp = json.loads((path / ".zattrs").read_text())
        ms = grp["multiscales"][0]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        shape = json.loads((path / "0" / ".zarray").read_text())["shape"]
    return shape, scale


# ---------------------------------------------------------------------------
# Patch positioning
# ---------------------------------------------------------------------------

def region_bbox_fr(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    return (min(zs), min(ys), min(xs)), (max(zs) + 1, max(ys) + 1, max(xs) + 1)


def center_patch_in_region(
    region_lo_fr, region_hi_fr, fr_scale_mm, r4_scale_mm,
) -> Dict[str, list]:
    """Centre a 128^3 4um patch in the region's bbox, snap to integer 4um voxels,
    and ensure it fits inside the region. Returns both 4um and fullres voxel boxes
    and the matching physical bbox."""
    # Region bbox in 4um voxel coords (resampled grid origin = full-res grid origin)
    z_ratio = r4_scale_mm[1] / fr_scale_mm[1]   # 1.0
    y_ratio = r4_scale_mm[2] / fr_scale_mm[2]   # 2.222...
    x_ratio = r4_scale_mm[3] / fr_scale_mm[3]   # 2.222...
    lo4 = (region_lo_fr[0] / z_ratio, region_lo_fr[1] / y_ratio, region_lo_fr[2] / x_ratio)
    hi4 = (region_hi_fr[0] / z_ratio, region_hi_fr[1] / y_ratio, region_hi_fr[2] / x_ratio)

    P = PATCH_VOX_4UM
    # Center of region in 4um voxel coords
    cz = (lo4[0] + hi4[0]) / 2.0
    cy = (lo4[1] + hi4[1]) / 2.0
    cx = (lo4[2] + hi4[2]) / 2.0
    z0_4 = int(round(cz - P / 2)); z1_4 = z0_4 + P
    y0_4 = int(round(cy - P / 2)); y1_4 = y0_4 + P
    x0_4 = int(round(cx - P / 2)); x1_4 = x0_4 + P

    # Verify fully inside region
    if not (lo4[0] <= z0_4 and z1_4 <= hi4[0] and
            lo4[1] <= y0_4 and y1_4 <= hi4[1] and
            lo4[2] <= x0_4 and x1_4 <= hi4[2]):
        raise RuntimeError(
            f"patch not contained in region bbox; "
            f"region4um=({lo4} -> {hi4}), patch4um=("
            f"({z0_4},{y0_4},{x0_4}) -> ({z1_4},{y1_4},{x1_4}))"
        )

    # Fullres voxel box (same physical extent)
    fz0 = int(round(z0_4 * z_ratio)); fz1 = fz0 + int(round(P * z_ratio))
    fy0 = int(round(y0_4 * y_ratio)); fy1 = fy0 + int(round(P * y_ratio))
    fx0 = int(round(x0_4 * x_ratio)); fx1 = fx0 + int(round(P * x_ratio))

    # Physical origin in mm (using 4um scale)
    phys_origin_mm = [z0_4 * r4_scale_mm[1], y0_4 * r4_scale_mm[2], x0_4 * r4_scale_mm[3]]
    phys_size_mm = [P * r4_scale_mm[1], P * r4_scale_mm[2], P * r4_scale_mm[3]]

    return {
        "box_4um_zyx_start": [z0_4, y0_4, x0_4],
        "box_4um_zyx_end":   [z1_4, y1_4, x1_4],
        "box_fr_zyx_start":  [fz0, fy0, fx0],
        "box_fr_zyx_end":    [fz1, fy1, fx1],
        "phys_origin_mm_zyx": phys_origin_mm,
        "phys_size_mm_zyx":   phys_size_mm,
    }


# ---------------------------------------------------------------------------
# Sphere rasterisation (anisotropic voxels, physical-radius sphere)
# ---------------------------------------------------------------------------

def rasterise_spheres(
    centroids_local_fr,   # iterable of (z, y, x) in fullres-patch-local voxel coords
    fr_patch_shape,       # (Z, Y, X) in fullres voxels
    fr_voxel_mm,          # (z_mm, y_mm, x_mm)
    radius_um=SPHERE_RADIUS_UM,
) -> np.ndarray:
    Z, Y, X = fr_patch_shape
    out = np.zeros(fr_patch_shape, dtype=np.uint8)
    rz_um = radius_um
    z_mm, y_mm, x_mm = fr_voxel_mm
    # voxel-radius envelopes per axis (used only to bound the inner loop)
    rz_v = radius_um / (z_mm * 1000.0)
    ry_v = radius_um / (y_mm * 1000.0)
    rx_v = radius_um / (x_mm * 1000.0)
    for (zc, yc, xc) in centroids_local_fr:
        z_min = max(0, int(np.floor(zc - rz_v)))
        z_max = min(Z, int(np.ceil(zc + rz_v)) + 1)
        y_min = max(0, int(np.floor(yc - ry_v)))
        y_max = min(Y, int(np.ceil(yc + ry_v)) + 1)
        x_min = max(0, int(np.floor(xc - rx_v)))
        x_max = min(X, int(np.ceil(xc + rx_v)) + 1)
        if z_min >= z_max or y_min >= y_max or x_min >= x_max:
            continue
        zz = (np.arange(z_min, z_max) - zc) * z_mm * 1000.0
        yy = (np.arange(y_min, y_max) - yc) * y_mm * 1000.0
        xx = (np.arange(x_min, x_max) - xc) * x_mm * 1000.0
        d2 = (zz[:, None, None] ** 2 +
              yy[None, :, None] ** 2 +
              xx[None, None, :] ** 2)
        block = (d2 <= rz_um ** 2)
        out[z_min:z_max, y_min:y_max, x_min:x_max] |= block.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="sub-IBA1brain11")
    p.add_argument("--canvas-root", default="/nfs/khan/datasets/CANVAS/bids")
    p.add_argument(
        "--csv-dir",
        default="/nfs/khan/datasets/CANVAS/derivatives/csv_annotations/sub-IBA1brain11",
    )
    p.add_argument(
        "--gt-root",
        default=os.environ.get("GT", "/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized"),
    )
    args = p.parse_args()

    bids = Path(args.canvas_root)
    fullres_zarr = next((bids / args.subject / "micr").glob(f"{args.subject}_*SPIM.ome.zarr"))
    res4_zarr = next((bids / "derivatives" / "resampled" / args.subject / "micr").glob(
        f"{args.subject}_*res-4um*SPIM.ome.zarr"))

    fr_shape, fr_scale = read_zarr_meta(fullres_zarr, version="v0.5")
    r4_shape, r4_scale = read_zarr_meta(res4_zarr, version="v0.4")
    print(f"fullres: shape={fr_shape}  voxel mm (c,z,y,x)={fr_scale}")
    print(f"4um    : shape={r4_shape}  voxel mm (c,z,y,x)={r4_scale}")

    fr_arr = zarr.open(str(fullres_zarr / "s0"), mode="r")
    r4_arr = zarr.open(str(res4_zarr / "0"), mode="r")

    centroids_by_region = load_centroids(Path(args.csv_dir))
    print(f"regions: {sorted(centroids_by_region)}")

    out_root = Path(args.gt_root) / "Iba1" / args.subject
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"writing to: {out_root}")

    # Affines (in mm; matches the zarr's (z,y,x) data order, like the rest of
    # the Lumivox pipeline).
    affine_4um_iso = np.diag([0.004, 0.004, 0.004, 1.0])
    affine_fr = np.diag([fr_scale[1], fr_scale[2], fr_scale[3], 1.0])

    summary = []
    for tag in sorted(centroids_by_region):
        pts = centroids_by_region[tag]
        lo_fr, hi_fr = region_bbox_fr(pts)
        boxes = center_patch_in_region(lo_fr, hi_fr, fr_scale, r4_scale)
        print(f"\n[{tag}] n={len(pts)}  region_bbox(z,y,x): {lo_fr} -> {hi_fr}")
        print(f"  4um patch (z,y,x):    {boxes['box_4um_zyx_start']} -> {boxes['box_4um_zyx_end']}")
        print(f"  fullres patch (z,y,x):{boxes['box_fr_zyx_start']} -> {boxes['box_fr_zyx_end']}")

        # Extract 4um crop (128^3)
        z0,y0,x0 = boxes["box_4um_zyx_start"]; z1,y1,x1 = boxes["box_4um_zyx_end"]
        crop_4um = np.asarray(r4_arr[0, z0:z1, y0:y1, x0:x1], dtype=np.uint16)

        # Extract fullres crop (128 x 284 x 284)
        fz0,fy0,fx0 = boxes["box_fr_zyx_start"]; fz1,fy1,fx1 = boxes["box_fr_zyx_end"]
        crop_fr = np.asarray(fr_arr[0, fz0:fz1, fy0:fy1, fx0:fx1], dtype=np.uint16)

        # Rasterise sphere mask (fullres shape, 8um physical radius)
        in_patch_local = []
        for (xc, yc, zc) in pts:
            if fz0 <= zc < fz1 and fy0 <= yc < fy1 and fx0 <= xc < fx1:
                in_patch_local.append((zc - fz0, yc - fy0, xc - fx0))
        sphere_fr = rasterise_spheres(
            in_patch_local, crop_fr.shape,
            fr_voxel_mm=(fr_scale[1], fr_scale[2], fr_scale[3]),
        )
        print(f"  centroids inside patch: {len(in_patch_local)} / {len(pts)}  "
              f"(sphere fill: {100*sphere_fr.mean():.3f}% of voxels)")

        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(crop_4um, affine_4um_iso), str(out_dir / "crop_4um.nii.gz"))
        nib.save(nib.Nifti1Image(crop_fr, affine_fr), str(out_dir / "crop_fullres.nii.gz"))
        nib.save(nib.Nifti1Image(sphere_fr, affine_fr),
                 str(out_dir / "centroids_spheres_fullres.nii.gz"))

        meta = {
            "subject": args.subject,
            "region_tag": tag,
            "n_centroids_region_total": len(pts),
            "n_centroids_in_patch": len(in_patch_local),
            "patch": boxes,
            "voxel_size_mm": {
                "fullres_zyx": [fr_scale[1], fr_scale[2], fr_scale[3]],
                "4um_zyx":     [r4_scale[1], r4_scale[2], r4_scale[3]],
            },
            "sphere_radius_um": SPHERE_RADIUS_UM,
            "files": {
                "crop_4um":       str(out_dir / "crop_4um.nii.gz"),
                "crop_fullres":   str(out_dir / "crop_fullres.nii.gz"),
                "spheres_fullres":str(out_dir / "centroids_spheres_fullres.nii.gz"),
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        summary.append({"region": tag, **{k: meta[k] for k in (
            "n_centroids_region_total", "n_centroids_in_patch", "patch"
        )}})

    # Top-level summary
    (out_root / "summary.json").write_text(json.dumps({
        "subject": args.subject,
        "regions": summary,
        "sphere_radius_um": SPHERE_RADIUS_UM,
    }, indent=2))
    print(f"\nsummary -> {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
