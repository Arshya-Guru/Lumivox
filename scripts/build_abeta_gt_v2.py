"""Build the v2 Abeta fine-tuning ground-truth patch set.

Atlas-samples 100 fresh Abeta patches across all 6 SPIMquant datasets, stratified
40/30/20/10 across cortex/hippocampus/striatum/cerebellum (the original SSL
proportions). For each patch we save just two NIfTIs:

    $GT/Abeta/v2/{roi}/{patch_id}/
        crop_4um.nii.gz       # 128^3, 4 um isotropic
        crop_fullres.nii.gz   # anisotropic, same physical extent

Plus a top-level $GT/Abeta/v2/manifest.json with every patch's metadata, and
side-by-side axial QC PNGs under qc_images/abeta_v2_pairs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import zarr

# lumivox.data.* internals — discover_subjects + stain channel + manifest builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumivox.data.manifest import (
    discover_subjects,
    resolve_stain_channel,
)
from zarrnii import ZarrNii, ZarrNiiAtlas


PATCH_4UM = 128
RES_4UM_MM = 0.004
PATCH_PHYS_UM = PATCH_4UM * RES_4UM_MM * 1000.0  # 512 um

REGION_GROUPS = {
    "cortex":      (["L_Isocortex",                 "R_Isocortex"],                 0.40),
    "hippocampus": (["L_Hippocampal formation",     "R_Hippocampal formation"],     0.30),
    "striatum":    (["L_Striatum",                  "R_Striatum"],                  0.20),
    "cerebellum":  (["L_Cerebellum",                "R_Cerebellum"],                0.10),
}

ALL_DATASETS = [
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_vaccine_batch",
]


# ---------------------------------------------------------------------------
# Atlas sampling (modified copy of build_patch_manifest that TAGS each patch
# with its region_group — the original strips that info)
# ---------------------------------------------------------------------------

def sample_patches_stratified(
    dataset_roots: List[str],
    stain: str,
    region_groups: Dict,
    n_patches: int,
    seg_level: str = "roi22",
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.default_rng(seed)

    # Per-group target counts
    total_weight = sum(w for _, w in region_groups.values())
    group_targets: Dict[str, int] = {}
    allocated = 0
    group_names = list(region_groups.keys())
    for i, name in enumerate(group_names):
        _, weight = region_groups[name]
        if i == len(group_names) - 1:
            group_targets[name] = n_patches - allocated
        else:
            n = int(round(n_patches * weight / total_weight))
            group_targets[name] = n
            allocated += n
    print(f"Per-group targets: {group_targets}  (sum={sum(group_targets.values())})")

    # Discover subjects
    all_subjects: List[Dict[str, str]] = []
    for root in dataset_roots:
        subs = discover_subjects(root, seg_level=seg_level)
        all_subjects.extend(subs)
        print(f"  {root}: {len(subs)} subjects")
    if not all_subjects:
        raise RuntimeError(f"No subjects found across {dataset_roots}")
    n_subjects = len(all_subjects)
    print(f"Total subjects: {n_subjects}")

    patches: List[Dict] = []
    for subj_idx, subj in enumerate(all_subjects):
        atlas = ZarrNiiAtlas.from_files(
            dseg_path=subj["dseg_path"],
            labels_path=subj["labels_path"],
        )
        centers_with_group: List[tuple] = []

        # Per-group, per-subject count
        for group_name, target_n in group_targets.items():
            region_list = region_groups[group_name][0]
            n_this = target_n // n_subjects
            if subj_idx < (target_n % n_subjects):
                n_this += 1
            if n_this == 0:
                continue
            subj_seed = int(rng.integers(0, 2**31))
            try:
                grp_centers = atlas.sample_region_patches(
                    n_patches=n_this,
                    region_ids=region_list,
                    seed=subj_seed,
                )
            except Exception as exc:
                print(f"    {subj['subject_id']} {group_name}: sample failed ({exc})")
                continue
            for c in grp_centers:
                centers_with_group.append((c, group_name))
        if not centers_with_group:
            continue

        stain_ch = resolve_stain_channel(
            subj["zarr_path"], stain, subj.get("fullres_zarr"),
        )

        # Physical → resampled voxel via fullres affine + JSON sidecar scale
        fullres_zarr = subj.get("fullres_zarr")
        sidecar_path = Path(subj["zarr_path"] + ".json")
        if not (fullres_zarr and subj["zarr_source"] == "resampled" and sidecar_path.exists()):
            print(f"    {subj['subject_id']}: missing fullres or sidecar; skip")
            continue
        fullres_inv = ZarrNii.from_ome_zarr(fullres_zarr, channels=[0]).affine.invert()
        sc = json.loads(sidecar_path.read_text())["fullres_to_resampled"]["scale"]
        res_scale = np.array([sc["z"], sc["y"], sc["x"]])

        for center, group_name in centers_with_group:
            vox_fullres = fullres_inv @ np.array(center)
            vox_resampled = vox_fullres * res_scale
            patches.append({
                "subject_id": subj["subject_id"],
                "dataset_root": subj["dataset_root"],
                "dataset_name": Path(subj["dataset_root"]).name,
                "zarr_path": subj["zarr_path"],
                "fullres_zarr": fullres_zarr,
                "resampled_sidecar": str(sidecar_path),
                "stain_channel": stain_ch,
                "region_group": group_name,
                "center_phys": [float(c) for c in center],
                "center_vox": [float(v) for v in vox_resampled],
                "center_vox_fullres": [float(v) for v in vox_fullres],
                "fullres_to_resampled_scale": {"z": sc["z"], "y": sc["y"], "x": sc["x"]},
            })
        print(f"  {subj['subject_id']}: kept {len(centers_with_group)} patches "
              f"(stain={stain} -> ch {stain_ch})")

    # Shuffle so write order is stain-spread, not subject-grouped
    rng_py = np.random.default_rng(seed + 1)
    rng_py.shuffle(patches)
    return patches


# ---------------------------------------------------------------------------
# Extraction + QC
# ---------------------------------------------------------------------------

def open_zarr_array(zarr_path: Path):
    """Open level-0 of an OME-Zarr (handles both v0.4 's0' and v0.4 '0' layouts)."""
    p0 = zarr_path / "0"
    p1 = zarr_path / "s0"
    if p0.exists() and (p0 / ".zarray").exists():
        return zarr.open(str(p0), mode="r")
    if p1.exists():
        return zarr.open(str(p1), mode="r")
    return zarr.open(str(zarr_path), mode="r")[0]


def fullres_voxel_um(fullres_zarr: Path) -> List[float]:
    """Return [z, y, x] full-res voxel size in micrometres."""
    zattrs_path = fullres_zarr / ".zattrs"
    if zattrs_path.exists():
        meta = json.loads(zattrs_path.read_text())
        scale_mm = meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        # scale_mm is [c, z, y, x] in mm
        return [scale_mm[1] * 1000.0, scale_mm[2] * 1000.0, scale_mm[3] * 1000.0]
    raise RuntimeError(f"no .zattrs at {fullres_zarr} (only OME-Zarr v0.4 supported here)")


def extract_crops(entry: Dict) -> Dict:
    """Extract 4um + fullres crops as numpy arrays + return metadata."""
    # 4um — easy, just integer-slice 128^3 around center_vox
    cv = entry["center_vox"]
    z0 = int(round(cv[0] - PATCH_4UM / 2)); z1 = z0 + PATCH_4UM
    y0 = int(round(cv[1] - PATCH_4UM / 2)); y1 = y0 + PATCH_4UM
    x0 = int(round(cv[2] - PATCH_4UM / 2)); x1 = x0 + PATCH_4UM

    r4_arr = open_zarr_array(Path(entry["zarr_path"]))
    stain_ch = entry["stain_channel"]
    # Bounds-check 4um (rare but possible at brain edge)
    _, Z4, Y4, X4 = r4_arr.shape
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z4 or y1 > Y4 or x1 > X4:
        return {"skip_reason": f"4um out of bounds (box {(z0,y0,x0)}->{(z1,y1,x1)} vs shape {(Z4,Y4,X4)})"}
    crop_4um = np.asarray(r4_arr[stain_ch, z0:z1, y0:y1, x0:x1], dtype=np.uint16)

    # Fullres — convert center_vox -> fullres voxel coords, derive anisotropic box
    fr_zarr = Path(entry["fullres_zarr"])
    fr_um = fullres_voxel_um(fr_zarr)  # [z, y, x] um
    target_um = PATCH_PHYS_UM
    pz = int(round(target_um / fr_um[0]))
    py = int(round(target_um / fr_um[1]))
    px = int(round(target_um / fr_um[2]))
    cv_fr = entry["center_vox_fullres"]
    fz0 = int(round(cv_fr[0] - pz / 2)); fz1 = fz0 + pz
    fy0 = int(round(cv_fr[1] - py / 2)); fy1 = fy0 + py
    fx0 = int(round(cv_fr[2] - px / 2)); fx1 = fx0 + px

    fr_arr = open_zarr_array(fr_zarr)
    _, FZ, FY, FX = fr_arr.shape
    if fz0 < 0 or fy0 < 0 or fx0 < 0 or fz1 > FZ or fy1 > FY or fx1 > FX:
        return {"skip_reason": f"fullres out of bounds (box {(fz0,fy0,fx0)}->{(fz1,fy1,fx1)} vs shape {(FZ,FY,FX)})"}
    crop_fr = np.asarray(fr_arr[stain_ch, fz0:fz1, fy0:fy1, fx0:fx1], dtype=np.uint16)

    return {
        "crop_4um": crop_4um,
        "crop_fullres": crop_fr,
        "fullres_voxel_um": fr_um,
        "box_4um_zyx": [z0, y0, x0, z1, y1, x1],
        "box_fr_zyx": [fz0, fy0, fx0, fz1, fy1, fx1],
        "fr_patch_shape": [pz, py, px],
    }


def render_qc(crop_4um, crop_fr, out_path: Path, title: str):
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
    ap.add_argument("--n-patches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gt-root",
        default=os.environ.get("GT", "/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized"),
    )
    ap.add_argument("--qc-dir", default="qc_images/abeta_v2_pairs")
    args = ap.parse_args()

    print(f"Sampling {args.n_patches} Abeta patches stratified across "
          f"{list(REGION_GROUPS)} (seed={args.seed})\n")
    patches = sample_patches_stratified(
        dataset_roots=ALL_DATASETS,
        stain="Abeta",
        region_groups=REGION_GROUPS,
        n_patches=args.n_patches,
        seed=args.seed,
    )
    if len(patches) != args.n_patches:
        print(f"\nWARNING: got {len(patches)} patches, expected {args.n_patches}")
    from collections import Counter
    print(f"\nSampled distribution: {dict(Counter(p['region_group'] for p in patches))}")

    v2_root = Path(args.gt_root) / "Abeta" / "v2"
    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    affine_4um = np.diag([RES_4UM_MM, RES_4UM_MM, RES_4UM_MM, 1.0])

    # Assign sequential patch IDs within each ROI so directory names are
    # stable and predictable (cortex_001, cortex_002, ...).
    counts = {r: 0 for r in REGION_GROUPS}
    top_entries = []
    skipped = []
    for entry in patches:
        roi = entry["region_group"]
        counts[roi] += 1
        patch_id = f"{roi}_{counts[roi]:03d}_{entry['subject_id']}"
        out_dir = v2_root / roi / patch_id
        try:
            res = extract_crops(entry)
        except Exception as exc:
            print(f"  {patch_id}: EXTRACTION FAILED ({exc})")
            counts[roi] -= 1  # don't burn the index for failed patches
            skipped.append({"patch_id": patch_id, **entry, "skip_reason": str(exc)})
            continue
        if "skip_reason" in res:
            print(f"  {patch_id}: skip ({res['skip_reason']})")
            counts[roi] -= 1
            skipped.append({"patch_id": patch_id, **entry, "skip_reason": res["skip_reason"]})
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        # only two files per patch dir, per the user's spec
        nib.save(nib.Nifti1Image(res["crop_4um"], affine_4um),
                 str(out_dir / "crop_4um.nii.gz"))
        fr_um = res["fullres_voxel_um"]
        affine_fr = np.diag([fr_um[0] / 1000.0, fr_um[1] / 1000.0, fr_um[2] / 1000.0, 1.0])
        nib.save(nib.Nifti1Image(res["crop_fullres"], affine_fr),
                 str(out_dir / "crop_fullres.nii.gz"))

        # QC PNG
        ds_short = entry["dataset_name"].replace("mouse_app_", "")
        render_qc(
            res["crop_4um"], res["crop_fullres"],
            qc_dir / f"{patch_id}.png",
            title=f"{ds_short}/{entry['subject_id']}  {patch_id}  ROI={roi}  "
                  f"(same {int(PATCH_PHYS_UM)}^3 um box)",
        )

        top_entries.append({
            "patch_id": patch_id,
            "roi": roi,
            "subject_id": entry["subject_id"],
            "dataset_name": entry["dataset_name"],
            "dataset_root": entry["dataset_root"],
            "stain_channel": entry["stain_channel"],
            "center_phys_mm_zyx": entry["center_phys"],
            "center_vox_4um_zyx": entry["center_vox"],
            "center_vox_fullres_zyx": entry["center_vox_fullres"],
            "box_4um_zyx": res["box_4um_zyx"],
            "box_fr_zyx": res["box_fr_zyx"],
            "fullres_voxel_um_zyx": res["fullres_voxel_um"],
            "fullres_zarr": entry["fullres_zarr"],
            "resampled_zarr": entry["zarr_path"],
            "resampled_sidecar": entry["resampled_sidecar"],
            "files": {
                "crop_4um":     str(out_dir / "crop_4um.nii.gz"),
                "crop_fullres": str(out_dir / "crop_fullres.nii.gz"),
            },
        })

    # Top-level manifest at v2 root (only sits at v2/, not inside any patch dir)
    v2_root.mkdir(parents=True, exist_ok=True)
    final_counts = dict(Counter(p["roi"] for p in top_entries))
    manifest = {
        "config": {
            "stain": "Abeta",
            "n_patches_requested": args.n_patches,
            "n_patches_written":   len(top_entries),
            "n_patches_skipped":   len(skipped),
            "seed": args.seed,
            "patch_size_4um_vox": PATCH_4UM,
            "patch_size_physical_um": PATCH_PHYS_UM,
            "region_groups": {n: {"regions": r, "weight": w}
                              for n, (r, w) in REGION_GROUPS.items()},
            "dataset_roots": ALL_DATASETS,
            "final_counts": final_counts,
        },
        "patches": top_entries,
        "skipped": skipped,
    }
    (v2_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(top_entries)} patches; {len(skipped)} skipped")
    print(f"  Final counts: {final_counts}")
    print(f"  GT root: {v2_root}")
    print(f"  Manifest: {v2_root/'manifest.json'}")
    print(f"  QC:       {qc_dir}/")


if __name__ == "__main__":
    main()
