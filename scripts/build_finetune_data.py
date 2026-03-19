"""Build fine-tuning data: 50 patches per stain with QC figures + saved crops.

For each patch:
  - QC figure (atlas zooms + zarr zooms + 128³ crop slices)
  - 128³ crop saved as .nii.gz
  - 128³ crop saved as .ome.zarr

Output structure:
  ft/{stain}/{dataset}/{subject}/{subject}_patch{N}_qc.png
  ft/{stain}/{dataset}/{subject}/{subject}_patch{N}_crop128.nii.gz
  ft/{stain}/{dataset}/{subject}/{subject}_patch{N}_crop128.ome.zarr

Usage:
    pixi run python scripts/build_finetune_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np

from zarrnii import ZarrNii, ZarrNiiAtlas
from lumivox.data.manifest import (
    discover_subjects,
    resolve_stain_channel,
    PREFERRED_SPIMQUANT,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REGION_GROUPS = {
    "cortex": (["L_Isocortex", "R_Isocortex"], 0.40),
    "hippocampus": (["L_Hippocampal formation", "R_Hippocampal formation"], 0.30),
    "striatum": (["L_Striatum", "R_Striatum"], 0.20),
    "cerebellum": (["L_Cerebellum", "R_Cerebellum"], 0.10),
}

ALL_DATASETS = [
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_vaccine_batch",
]

KI3_DATASETS = [
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch3",
]

N_PATCHES = 50
PATCH_SIZE = 256  # region read from zarr for context
CROP_SIZE = 128   # actual saved crop
SEED = 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distribute_patches(n_patches, n_subjects, region_groups, rng):
    """Compute per-subject, per-region patch counts."""
    # First distribute across regions by weight
    group_totals = {}
    allocated = 0
    groups = list(region_groups.keys())
    for i, name in enumerate(groups):
        _, weight = region_groups[name]
        total_weight = sum(w for _, w in region_groups.values())
        if i == len(groups) - 1:
            n = n_patches - allocated
        else:
            n = int(round(n_patches * weight / total_weight))
        group_totals[name] = n
        allocated += n

    # Then distribute each group's patches across subjects
    assignments = []  # list of (subject_idx, group_name, region_list)
    for group_name, n_group in group_totals.items():
        region_list = region_groups[group_name][0]
        for i in range(n_group):
            subj_idx = i % n_subjects
            assignments.append((subj_idx, group_name, region_list))

    rng.shuffle(assignments)
    return assignments, group_totals


def generate_qc_and_crop(
    subj, center_phys, center_vox, stain_ch, crop_size, patch_size,
    atlas, dseg_data, dseg_masked, target_mask,
    darr, output_dir, sub_id, dataset_name, patch_idx,
):
    """Generate QC figure + save 128³ crop as nii.gz and ome.zarr."""
    cz, cy, cx = int(round(center_vox[0])), int(round(center_vox[1])), int(round(center_vox[2]))

    # Dseg voxel coords for atlas overlay
    inv_affine = atlas.dseg.affine.invert()
    vox_dseg = inv_affine @ np.array(center_phys)
    di, dj, dk = int(round(vox_dseg[0])), int(round(vox_dseg[1])), int(round(vox_dseg[2]))

    label_val = dseg_data[di, dj, dk]
    label_name = atlas.labels_df[atlas.labels_df["index"] == int(label_val)]["name"].values
    label_str = label_name[0] if len(label_name) > 0 else f"label={int(label_val)}"

    # Extract the 128³ crop directly
    half_c = crop_size // 2
    cz0 = max(0, cz - half_c)
    cz1 = min(darr.shape[1], cz + half_c)
    cy0 = max(0, cy - half_c)
    cy1 = min(darr.shape[2], cy + half_c)
    cx0 = max(0, cx - half_c)
    cx1 = min(darr.shape[3], cx + half_c)
    crop = darr[0, cz0:cz1, cy0:cy1, cx0:cx1].compute().astype(np.float32)

    # Pad if near boundary
    if crop.shape != (crop_size, crop_size, crop_size):
        padded = np.zeros((crop_size, crop_size, crop_size), dtype=np.float32)
        padded[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        crop = padded

    crop_vmin, crop_vmax = np.percentile(crop, [1, 99])

    # Also read the 256 patch context for QC visualization
    half_p = patch_size // 2
    pz0 = max(0, cz - half_p)
    pz1 = min(darr.shape[1], cz + half_p)
    py0 = max(0, cy - half_p)
    py1 = min(darr.shape[2], cy + half_p)
    px0 = max(0, cx - half_p)
    px1 = min(darr.shape[3], cx + half_p)

    # --- Save crop as NIfTI ---
    prefix = f"{sub_id}_patch{patch_idx:02d}"
    out_dir = output_dir / dataset_name / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_path = out_dir / f"{prefix}_crop{crop_size}.nii.gz"
    # 4µm isotropic affine
    affine_4um = np.diag([0.004, 0.004, 0.004, 1.0])
    nii_img = nib.Nifti1Image(crop, affine_4um)
    nib.save(nii_img, str(nii_path))

    # --- Save crop as OME-Zarr ---
    zarr_out_path = out_dir / f"{prefix}_crop{crop_size}.ome.zarr"
    crop_zn = ZarrNii.from_darr(
        crop[np.newaxis],  # add channel dim (1, Z, Y, X)
        spacing=[0.004, 0.004, 0.004],  # 4µm isotropic in mm (z, y, x)
    )
    crop_zn.to_ome_zarr(str(zarr_out_path), max_layer=0)

    # --- QC Figure ---
    atlas_pads = [None, 60, 30, 15]
    fig, axes = plt.subplots(5, 4, figsize=(26, 32))

    # Row 0-2: Atlas 3 ortho × 4 zooms
    for row, (slice_data, slice_mask, dot_x, dot_y, dim_label, dim_val) in enumerate([
        (dseg_masked[di], target_mask[di], dk, dj, "dim0", di),
        (dseg_masked[:, dj, :], target_mask[:, dj, :], dk, di, "dim1", dj),
        (dseg_masked[:, :, dk], target_mask[:, :, dk], dj, di, "dim2", dk),
    ]):
        for col, pad in enumerate(atlas_pads):
            ax = axes[row, col]
            ax.imshow(slice_data, cmap="nipy_spectral", vmin=1, vmax=22, origin="lower")
            ax.contour(slice_mask, colors="lime", linewidths=0.8)
            ax.plot(dot_x, dot_y, "ro", markersize=10, markeredgecolor="white", markeredgewidth=2)
            if pad is not None:
                ax.set_xlim(dot_x - pad, dot_x + pad)
                ax.set_ylim(dot_y - pad, dot_y + pad)
            ax.set_title(f"{dim_label}={dim_val}" + (f" ±{pad}" if pad else " full"))

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
        ax.imshow(slab, cmap="gray", vmin=crop_vmin, vmax=crop_vmax, origin="lower",
                  extent=[rx0, rx1, ry0, ry1])
        # Show both patch box (red) and crop box (cyan)
        rect_p = Rectangle((px0, py0), px1-px0, py1-py0, lw=1.5, edgecolor="red", facecolor="none", ls="--")
        rect_c = Rectangle((cx0, cy0), cx1-cx0, cy1-cy0, lw=2, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect_p)
        ax.add_patch(rect_c)
        ax.set_title(f"z={cz} — {label} (red=256, cyan=128)")

    # Row 4: 1.2x zoom + 3 crop slices
    slab = darr[0, cz, max(0, cy-160):min(darr.shape[2], cy+160),
                max(0, cx-160):min(darr.shape[3], cx+160)].compute().astype(np.float32)
    ax = axes[4, 0]
    ax.imshow(slab, cmap="gray", vmin=crop_vmin, vmax=crop_vmax, origin="lower",
              extent=[max(0, cx-160), min(darr.shape[3], cx+160),
                      max(0, cy-160), min(darr.shape[2], cy+160)])
    rect_c = Rectangle((cx0, cy0), cx1-cx0, cy1-cy0, lw=2, edgecolor="cyan", facecolor="none")
    ax.add_patch(rect_c)
    ax.set_title(f"z={cz} — 1.2x (cyan=128³ crop)")

    mid = crop.shape[0] // 2
    for i, offset in enumerate([-10, 0, 10]):
        ax = axes[4, i + 1]
        sl = mid + offset
        if 0 <= sl < crop.shape[0]:
            ax.imshow(crop[sl], cmap="gray",
                      vmin=np.percentile(crop[sl], 0.5), vmax=np.percentile(crop[sl], 99.5))
        ax.set_title(f"Crop z={sl}")
        ax.axis("off")

    fig.suptitle(
        f"{dataset_name} / {sub_id} / patch {patch_idx} | vox ({cz},{cy},{cx}) | {label_str}\n"
        f"128³ crop saved | red=256 context, cyan=128 crop",
        fontsize=14,
    )
    plt.tight_layout()
    qc_path = out_dir / f"{prefix}_qc.png"
    fig.savefig(qc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return nii_path, zarr_out_path, qc_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_for_stain(stain, dataset_roots, output_base):
    print(f"\n{'='*60}")
    print(f"Building fine-tune data: {stain}, {N_PATCHES} patches")
    print(f"{'='*60}")

    rng = np.random.default_rng(SEED)

    # Discover all subjects
    all_subjects = []
    for ds in dataset_roots:
        subs = discover_subjects(ds)
        # Filter to resampled + sidecar
        subs = [s for s in subs if s["zarr_source"] == "resampled"
                and Path(s["zarr_path"] + ".json").exists()]
        all_subjects.extend(subs)
        print(f"  {Path(ds).name}: {len(subs)} subjects")

    n_subjects = len(all_subjects)
    print(f"  Total: {n_subjects} subjects")

    # Distribute patches
    assignments, group_totals = distribute_patches(N_PATCHES, n_subjects, REGION_GROUPS, rng)
    print(f"  Region distribution: {group_totals}")

    # Group assignments by subject
    from collections import defaultdict
    subj_assignments = defaultdict(list)
    for subj_idx, group_name, region_list in assignments:
        subj_assignments[subj_idx].append((group_name, region_list))

    output_dir = Path(output_base) / stain
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_count = 0
    manifest_entries = []

    for subj_idx, tasks in sorted(subj_assignments.items()):
        subj = all_subjects[subj_idx]
        sub_id = subj["subject_id"]
        dataset_name = Path(subj["dataset_root"]).name

        # Load atlas
        atlas = ZarrNiiAtlas.from_files(dseg_path=subj["dseg_path"], labels_path=subj["labels_path"])
        dseg_data = atlas.dseg.darr.compute().squeeze()
        target_ids = {3, 4, 5, 6, 9, 10, 21, 22}
        target_mask = np.isin(dseg_data, list(target_ids))
        dseg_masked = np.ma.masked_where(dseg_data == 0, dseg_data)

        # Resolve channel
        stain_ch = resolve_stain_channel(subj["zarr_path"], stain, subj.get("fullres_zarr"))

        # Load zarr
        znimg = ZarrNii.from_ome_zarr(subj["zarr_path"], channels=[stain_ch])
        darr = znimg.darr

        # Full-res affine + sidecar for voxel conversion
        znimg_full = ZarrNii.from_ome_zarr(subj["fullres_zarr"], channels=[0])
        fullres_inv = znimg_full.affine.invert()
        with open(subj["zarr_path"] + ".json") as f:
            sc = json.load(f)["fullres_to_resampled"]["scale"]
        res_scale = np.array([sc["z"], sc["y"], sc["x"]])

        local_patch_idx = 0
        for group_name, region_list in tasks:
            # Sample 1 center from this region group
            center_seed = int(rng.integers(0, 2**31))
            centers = atlas.sample_region_patches(
                n_patches=1, region_ids=region_list, seed=center_seed,
            )
            if not centers:
                print(f"    SKIP {sub_id} {group_name}: no centers sampled")
                continue

            center = centers[0]
            vox_fullres = fullres_inv @ np.array(center)
            center_vox = vox_fullres * res_scale

            nii_path, zarr_path, qc_path = generate_qc_and_crop(
                subj, list(center), list(center_vox), stain_ch,
                CROP_SIZE, PATCH_SIZE,
                atlas, dseg_data, dseg_masked, target_mask,
                darr, output_dir, sub_id, dataset_name, local_patch_idx,
            )

            manifest_entries.append({
                "subject_id": sub_id,
                "dataset": dataset_name,
                "region_group": group_name,
                "center_phys": [float(c) for c in center],
                "center_vox": [float(v) for v in center_vox],
                "stain_channel": stain_ch,
                "nii_path": str(nii_path),
                "zarr_path": str(zarr_path),
                "qc_path": str(qc_path),
            })

            patch_count += 1
            local_patch_idx += 1
            print(f"  [{patch_count}/{N_PATCHES}] {dataset_name}/{sub_id} patch{local_patch_idx-1:02d} ({group_name})")

    # Save manifest
    manifest = {
        "config": {
            "stain": stain,
            "n_patches": patch_count,
            "crop_size": CROP_SIZE,
            "patch_size": PATCH_SIZE,
            "region_groups": {name: {"regions": r, "weight": w} for name, (r, w) in REGION_GROUPS.items()},
            "seed": SEED,
        },
        "patches": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved manifest: {manifest_path}")
    print(f"  Total patches: {patch_count}")


if __name__ == "__main__":
    build_for_stain("Abeta", ALL_DATASETS, "ft")
    build_for_stain("Iba1", KI3_DATASETS, "ft")
    print("\nAll done!")
