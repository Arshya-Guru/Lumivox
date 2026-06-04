"""Promote a hand-curated subset of the old ki3 Iba1 patches to the new GT
location, including a matching full-res NIfTI for each.

Samples N (default 4) patches per ROI from old/Iba1/manifest.json, then for
each:
  * copies the existing 4um isotropic 128^3 nifti into the GT tree
  * discovers the source resampled zarr's JSON sidecar (which records the
    fullres path and the fullres->resampled scale factors)
  * uses center_vox (recorded by the old build script in resampled voxel space)
    to carve out an anisotropic full-res box covering the same physical
    512x512x512 um extent, and saves it as a NIfTI with the correct affine
  * writes a per-patch meta.json (ROI, center coords, voxel sizes, source paths)
  * writes a side-by-side axial-only QC PNG

Output:
  $GT/Iba1/myiba1_patches/{roi}/{patch_id}_crop_4um.nii.gz
  $GT/Iba1/myiba1_patches/{roi}/{patch_id}_crop_fullres.nii.gz
  $GT/Iba1/myiba1_patches/{roi}/{patch_id}_meta.json
  $GT/Iba1/myiba1_patches/manifest.json
  qc_images/myiba1_pairs/{roi}__{patch_id}.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import zarr


PATCH_4UM = 128
RES_4UM_MM = 0.004
ROIS = ("cortex", "hippocampus", "striatum", "cerebellum")


# ---------------------------------------------------------------------------
# Per-subject resampled zarr discovery (inlined so this script doesn't pull in
# lumivox.data.* and its numba/blosc2 transitive deps)
# ---------------------------------------------------------------------------

def find_resampled_sidecar(prado_root: Path, dataset: str, subject: str) -> Optional[Path]:
    micr = prado_root / dataset / "bids" / "derivatives" / "resampled" / subject / "micr"
    if not micr.exists():
        return None
    candidates = sorted(micr.glob(f"{subject}_*res-4um*.ome.zarr.json"))
    # Mirror lumivox.data.manifest._prefer_standard_acq
    standard = [p for p in candidates if "45deg" not in p.name and "90deg" not in p.name]
    pool = standard or candidates
    if not pool:
        return None
    # Prefer plain acq-imaris4x over imaris4x166 / other variants
    plain = [p for p in pool if "_acq-imaris4x_" in p.name]
    return (plain or pool)[0]


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_per_roi(
    patches: List[Dict],
    per_roi: int,
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    by_roi: Dict[str, List[Dict]] = {r: [] for r in ROIS}
    for p in patches:
        by_roi.setdefault(p["region_group"], []).append(p)
    sampled: Dict[str, List[Dict]] = {}
    for roi in ROIS:
        avail = by_roi.get(roi, [])
        if len(avail) < per_roi:
            raise RuntimeError(f"ROI {roi}: only {len(avail)} patches available (need {per_roi})")
        sampled[roi] = sorted(rng.sample(avail, per_roi),
                              key=lambda e: (e["dataset"], e["subject_id"]))
    return sampled


# ---------------------------------------------------------------------------
# Fullres extraction
# ---------------------------------------------------------------------------

def extract_fullres_crop(
    sidecar_path: Path,
    center_vox_4um: List[float],
    stain_channel: int,
) -> Tuple[np.ndarray, Dict]:
    """Carve a fullres box covering 128*4=512 um per side.

    Returns (data array shaped (Z,Y,X), info dict with voxel sizes and box).
    """
    sc = json.loads(sidecar_path.read_text())
    fr_path = Path(sc["source"]["path"])
    fr_um = sc["source"]["voxel_sizes_um"]   # {'z':2.2, 'y':1.625, 'x':1.625}
    scale = sc["fullres_to_resampled"]["scale"]  # voxel_resampled = voxel_fullres * scale
    fr_shape = sc["source"]["shape_czyx"]

    # Map resampled (4um) voxel coord -> fullres voxel coord
    zc_4 = float(center_vox_4um[0]); yc_4 = float(center_vox_4um[1]); xc_4 = float(center_vox_4um[2])
    zc_fr = zc_4 / scale["z"]
    yc_fr = yc_4 / scale["y"]
    xc_fr = xc_4 / scale["x"]

    # Patch dimensions in fullres voxels (cover 512 um per axis)
    target_um = PATCH_4UM * RES_4UM_MM * 1000.0  # 512 um
    pz = int(round(target_um / fr_um["z"]))
    py = int(round(target_um / fr_um["y"]))
    px = int(round(target_um / fr_um["x"]))

    z0 = int(round(zc_fr - pz / 2)); z1 = z0 + pz
    y0 = int(round(yc_fr - py / 2)); y1 = y0 + py
    x0 = int(round(xc_fr - px / 2)); x1 = x0 + px

    # Clamp into volume bounds (shape is c,z,y,x); shift if needed to keep full size
    _, Z, Y, X = fr_shape
    def clamp(lo, hi, full):
        if lo < 0: hi -= lo; lo = 0
        if hi > full: lo -= (hi - full); hi = full
        return max(0, lo), min(full, hi)
    z0, z1 = clamp(z0, z1, Z)
    y0, y1 = clamp(y0, y1, Y)
    x0, x1 = clamp(x0, x1, X)

    # The fullres zarr is v0.4 (zarr v2). Channel-axis dataset is at "0".
    arr = zarr.open(str(fr_path / "0"), mode="r")
    data = np.asarray(arr[stain_channel, z0:z1, y0:y1, x0:x1], dtype=np.uint16)

    info = {
        "fullres_zarr": str(fr_path),
        "fullres_voxel_um_zyx": [fr_um["z"], fr_um["y"], fr_um["x"]],
        "fullres_box_zyx_start": [z0, y0, x0],
        "fullres_box_zyx_end":   [z1, y1, x1],
        "fullres_box_shape_zyx": list(data.shape),
        "fullres_to_resampled_scale_zyx": [scale["z"], scale["y"], scale["x"]],
        "center_vox_fullres_zyx": [zc_fr, yc_fr, xc_fr],
    }
    return data, info


# ---------------------------------------------------------------------------
# QC PNG: axial mid-slice 4um  vs  axial mid-slice fullres
# ---------------------------------------------------------------------------

def render_qc(crop_4um: np.ndarray, crop_fr: np.ndarray, out_path: Path, title: str):
    z4 = crop_4um.shape[0] // 2
    zf = crop_fr.shape[0] // 2
    vmax_4 = float(np.percentile(crop_4um, 99.5)) or 1.0
    vmax_f = float(np.percentile(crop_fr, 99.5)) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    axes[0].imshow(crop_4um[z4], cmap="gray", vmin=0, vmax=vmax_4)
    axes[0].set_title(f"4um iso axial  z={z4}/{crop_4um.shape[0]}  shape={crop_4um.shape}")
    axes[1].imshow(crop_fr[zf], cmap="gray", vmin=0, vmax=vmax_f)
    axes[1].set_title(f"fullres axial  z={zf}/{crop_fr.shape[0]}  shape={crop_fr.shape}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-manifest", default="old/Iba1/manifest.json")
    ap.add_argument("--prado-root", default="/nfs/trident3/lightsheet/prado")
    ap.add_argument(
        "--gt-root",
        default=os.environ.get("GT", "/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized"),
    )
    ap.add_argument("--qc-dir", default="qc_images/myiba1_pairs")
    ap.add_argument("--per-roi", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    old_manifest_path = Path(args.old_manifest)
    old_manifest = json.loads(old_manifest_path.read_text())
    sampled = sample_per_roi(old_manifest["patches"], args.per_roi, args.seed)

    gt_root = Path(args.gt_root) / "Iba1" / "myiba1_patches"
    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    affine_4um = np.diag([RES_4UM_MM, RES_4UM_MM, RES_4UM_MM, 1.0])

    top_entries = []
    print(f"Sampling {args.per_roi} patches/ROI from {old_manifest_path} (seed={args.seed})\n")
    for roi in ROIS:
        out_roi = gt_root / roi
        out_roi.mkdir(parents=True, exist_ok=True)
        print(f"[{roi}]  {len(sampled[roi])} patches:")
        for entry in sampled[roi]:
            ds = entry["dataset"]; sub = entry["subject_id"]
            pid = Path(entry["nii_path"]).name.replace("_crop128.nii.gz", "")  # sub-X_patchNN
            print(f"  - {ds:<40s} {sub:<14s} {pid}")

            # Copy 4um nifti
            src_4um = Path(entry["nii_path"])
            if not src_4um.exists():
                src_4um = old_manifest_path.parent / f"{ds}/{sub}/{pid}_crop128.nii.gz"
            if not src_4um.exists():
                # Fall back to old/Iba1 (we moved ft/Iba1 there)
                src_4um = Path("old") / "Iba1" / ds / sub / f"{pid}_crop128.nii.gz"
            dst_4um = out_roi / f"{pid}_crop_4um.nii.gz"
            shutil.copyfile(src_4um, dst_4um)
            crop_4um = nib.load(str(dst_4um)).get_fdata().astype(np.uint16)

            # Discover resampled sidecar for this subject
            sidecar = find_resampled_sidecar(Path(args.prado_root), ds, sub)
            if sidecar is None:
                raise RuntimeError(f"no resampled sidecar found for {ds}/{sub}")

            # Extract fullres
            stain_ch = int(entry["stain_channel"])
            crop_fr, fr_info = extract_fullres_crop(
                sidecar, entry["center_vox"], stain_channel=stain_ch,
            )
            affine_fr = np.diag([
                fr_info["fullres_voxel_um_zyx"][0] / 1000.0,
                fr_info["fullres_voxel_um_zyx"][1] / 1000.0,
                fr_info["fullres_voxel_um_zyx"][2] / 1000.0,
                1.0,
            ])
            dst_fr = out_roi / f"{pid}_crop_fullres.nii.gz"
            nib.save(nib.Nifti1Image(crop_fr, affine_fr), str(dst_fr))

            # meta.json
            meta = {
                "patch_id": pid,
                "roi": roi,
                "dataset": ds,
                "subject_id": sub,
                "stain": "Iba1",
                "stain_channel": stain_ch,
                "source_old_4um_path": str(src_4um),
                "center_phys_mm_zyx": entry["center_phys"],
                "center_vox_4um_zyx": entry["center_vox"],
                "resampled_sidecar": str(sidecar),
                **fr_info,
                "voxel_size_mm": {
                    "4um_zyx":     [RES_4UM_MM, RES_4UM_MM, RES_4UM_MM],
                    "fullres_zyx": list(affine_fr.diagonal()[:3]),
                },
                "files": {
                    "crop_4um":     str(dst_4um),
                    "crop_fullres": str(dst_fr),
                },
            }
            (out_roi / f"{pid}_meta.json").write_text(json.dumps(meta, indent=2))
            top_entries.append({
                "patch_id": pid, "roi": roi, "dataset": ds, "subject_id": sub,
                **{k: meta[k] for k in (
                    "stain_channel", "center_phys_mm_zyx", "center_vox_4um_zyx",
                    "fullres_box_zyx_start", "fullres_box_zyx_end", "fullres_box_shape_zyx",
                    "fullres_voxel_um_zyx", "files",
                )},
            })

            # QC PNG
            title = f"{ds}/{sub} {pid}  ROI={roi}  (same 512x512x512 um box)"
            render_qc(crop_4um, crop_fr, qc_dir / f"{roi}__{pid}.png", title)

    manifest = {
        "config": {
            "stain": "Iba1",
            "per_roi": args.per_roi,
            "seed": args.seed,
            "source_manifest": str(old_manifest_path),
            "patch_4um_shape_zyx": [PATCH_4UM, PATCH_4UM, PATCH_4UM],
            "patch_physical_size_um": PATCH_4UM * RES_4UM_MM * 1000.0,
            "gt_root": str(gt_root),
        },
        "patches": top_entries,
    }
    out_manifest = gt_root / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(top_entries)} patches across {len(ROIS)} ROIs")
    print(f"  manifest: {out_manifest}")
    print(f"  qc:       {qc_dir}/")


if __name__ == "__main__":
    main()
