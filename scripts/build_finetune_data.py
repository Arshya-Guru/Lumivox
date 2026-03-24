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
from typing import Optional

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
    """Compute per-subject, per-region patch counts.

    Spreads patches across subjects as evenly as possible while respecting
    region weights.  Shuffles subject order per region group so different
    datasets get coverage even when n_patches < n_subjects.
    """
    # First distribute across regions by weight
    group_totals = {}
    allocated = 0
    groups = list(region_groups.keys())
    total_weight = sum(w for _, w in region_groups.values())
    for i, name in enumerate(groups):
        _, weight = region_groups[name]
        if i == len(groups) - 1:
            n = n_patches - allocated
        else:
            n = int(round(n_patches * weight / total_weight))
        group_totals[name] = n
        allocated += n

    # Spread each group's patches across a shuffled subject order
    # so all datasets get representation, not just the first alphabetically
    assignments = []
    subject_indices = list(range(n_subjects))
    for group_name, n_group in group_totals.items():
        region_list = region_groups[group_name][0]
        shuffled = subject_indices.copy()
        rng.shuffle(shuffled)
        for i in range(n_group):
            subj_idx = shuffled[i % n_subjects]
            assignments.append((subj_idx, group_name, region_list))

    rng.shuffle(assignments)
    return assignments, group_totals


def find_fieldfrac(spimquant_micr: Path, stain: str) -> Optional[Path]:
    """Find the subject-space fieldfrac NIfTI for a given stain.

    Tries otsu+k3i2 first, then th900. Skips template-space (space-ABAv3) files.
    """
    for method in ["otsu+k3i2", "th900"]:
        candidates = sorted(spimquant_micr.glob(
            f"*_stain-{stain}_level-5_desc-{method}_fieldfrac.nii.gz"
        ))
        # Filter out template-space versions
        candidates = [c for c in candidates if "space-ABAv3" not in c.name]
        if candidates:
            return candidates[0]
    # Also try case-insensitive stain matching
    for method in ["otsu+k3i2", "th900"]:
        for f in spimquant_micr.glob(f"*_level-5_desc-{method}_fieldfrac.nii.gz"):
            if "space-ABAv3" not in f.name and stain.lower() in f.name.lower():
                return f
    return None


def extract_seg_crop(
    fieldfrac_path: Path,
    center_phys: list,
    dseg_affine_inv,
    dseg_shape: tuple,
    crop_size: int,
) -> Optional[np.ndarray]:
    """Extract and upsample a seg mask crop from the level-5 fieldfrac NIfTI.

    The fieldfrac shares the same voxel grid as the dseg. We use the dseg
    affine to map from physical space to level-5 voxels, extract the region
    corresponding to the 128³ crop at 4µm, and upsample to 128³.
    """
    from scipy.ndimage import zoom as ndi_zoom

    ff_img = nib.load(str(fieldfrac_path))
    ff_data = ff_img.get_fdata().astype(np.float32)

    # Center in dseg/level-5 voxel space
    vox_center = dseg_affine_inv @ np.array(center_phys)

    # Crop extent in physical space: 128 * 4µm = 0.512mm per axis
    crop_extent_mm = crop_size * 0.004  # mm

    # Compute level-5 voxel sizes from dseg affine (absolute diagonal values)
    # dseg_affine_inv maps physical→voxel, so voxel_size = 1/|inv_diag|
    # But easier: just compute how many level-5 voxels span the crop
    # Use the dseg affine inverse to map two points and get the span
    corner_lo = dseg_affine_inv @ np.array([
        center_phys[0] - crop_extent_mm / 2,
        center_phys[1] - crop_extent_mm / 2,
        center_phys[2] - crop_extent_mm / 2,
    ])
    corner_hi = dseg_affine_inv @ np.array([
        center_phys[0] + crop_extent_mm / 2,
        center_phys[1] + crop_extent_mm / 2,
        center_phys[2] + crop_extent_mm / 2,
    ])

    # Get integer voxel bounds (handle axis flips from negative affine entries)
    for dim in range(3):
        if corner_lo[dim] > corner_hi[dim]:
            corner_lo[dim], corner_hi[dim] = corner_hi[dim], corner_lo[dim]

    i0 = max(0, int(np.floor(corner_lo[0])))
    i1 = min(ff_data.shape[0], int(np.ceil(corner_hi[0])) + 1)
    j0 = max(0, int(np.floor(corner_lo[1])))
    j1 = min(ff_data.shape[1], int(np.ceil(corner_hi[1])) + 1)
    k0 = max(0, int(np.floor(corner_lo[2])))
    k1 = min(ff_data.shape[2], int(np.ceil(corner_hi[2])) + 1)

    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return None

    seg_chunk = ff_data[i0:i1, j0:j1, k0:k1]

    if seg_chunk.size == 0:
        return None

    # Upsample to crop_size³ using nearest-neighbor (fieldfrac is continuous 0-1
    # but nearest avoids blurring the boundaries)
    zoom_factors = (
        crop_size / seg_chunk.shape[0],
        crop_size / seg_chunk.shape[1],
        crop_size / seg_chunk.shape[2],
    )
    seg_upsampled = ndi_zoom(seg_chunk, zoom_factors, order=0)

    # Ensure exact shape
    if seg_upsampled.shape != (crop_size, crop_size, crop_size):
        result = np.zeros((crop_size, crop_size, crop_size), dtype=np.float32)
        slices = tuple(slice(0, min(s, crop_size)) for s in seg_upsampled.shape)
        result[slices[0], slices[1], slices[2]] = seg_upsampled[slices[0], slices[1], slices[2]]
        seg_upsampled = result

    return seg_upsampled.astype(np.float32)


def generate_qc_and_crop(
    subj, center_phys, center_vox, stain_ch, crop_size, patch_size,
    atlas, dseg_data, dseg_masked, target_mask,
    darr, output_dir, sub_id, dataset_name, patch_idx,
    stain_name="Abeta",
    spimquant_micr=None,
):
    """Generate QC figure + save 128³ crop + seg mask as nii.gz."""
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

    # --- Extract seg mask if available ---
    seg_crop = None
    seg_path = None
    if spimquant_micr is not None:
        ff_path = find_fieldfrac(Path(spimquant_micr), stain_name)
        if ff_path is not None:
            seg_crop = extract_seg_crop(
                ff_path, center_phys, inv_affine, dseg_data.shape, crop_size,
            )

    # --- Save crop as NIfTI ---
    prefix = f"{sub_id}_patch{patch_idx:02d}"
    out_dir = output_dir / dataset_name / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_path = out_dir / f"{prefix}_crop{crop_size}.nii.gz"
    affine_4um = np.diag([0.004, 0.004, 0.004, 1.0])
    nii_img = nib.Nifti1Image(crop, affine_4um)
    nib.save(nii_img, str(nii_path))

    # Save seg mask if available
    if seg_crop is not None:
        seg_path = out_dir / f"{prefix}_seg{crop_size}.nii.gz"
        seg_img = nib.Nifti1Image(seg_crop, affine_4um)
        nib.save(seg_img, str(seg_path))

    # --- QC Figure ---
    has_seg = seg_crop is not None
    atlas_pads = [None, 60, 30, 15]
    fig, axes = plt.subplots(6 if has_seg else 5, 4, figsize=(26, 38 if has_seg else 32))

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
            # Overlay seg at 30% opacity if available
            if has_seg:
                seg_slice = seg_crop[sl]
                seg_overlay = np.ma.masked_where(seg_slice < 0.01, seg_slice)
                ax.imshow(seg_overlay, cmap="hot", alpha=0.3, vmin=0, vmax=1)
        ax.set_title(f"Crop z={sl}" + (" + seg" if has_seg else ""))
        ax.axis("off")

    # Row 5 (if seg available): 3 crop slices with seg only + 1 overlay detail
    if has_seg:
        for i, offset in enumerate([-10, 0, 10]):
            ax = axes[5, i]
            sl = mid + offset
            if 0 <= sl < crop.shape[0]:
                ax.imshow(seg_crop[sl], cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Seg z={sl} (fieldfrac)")
            ax.axis("off")
        # Last panel: overlay with stronger opacity
        ax = axes[5, 3]
        if 0 <= mid < crop.shape[0]:
            ax.imshow(crop[mid], cmap="gray",
                      vmin=np.percentile(crop[mid], 0.5), vmax=np.percentile(crop[mid], 99.5))
            seg_overlay = np.ma.masked_where(seg_crop[mid] < 0.01, seg_crop[mid])
            ax.imshow(seg_overlay, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"Overlay z={mid} (50% opacity)")
        ax.axis("off")

    seg_status = "seg=YES" if has_seg else "seg=N/A"
    fig.suptitle(
        f"{dataset_name} / {sub_id} / patch {patch_idx} | vox ({cz},{cy},{cx}) | {label_str}\n"
        f"128³ crop saved | red=256 context, cyan=128 crop | {seg_status}",
        fontsize=14,
    )
    plt.tight_layout()
    qc_path = out_dir / f"{prefix}_qc.png"
    fig.savefig(qc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return nii_path, seg_path, qc_path


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

            # spimquant micr dir (same dir as the dseg file)
            spimquant_micr = str(Path(subj["dseg_path"]).parent)

            nii_path, seg_path, qc_path = generate_qc_and_crop(
                subj, list(center), list(center_vox), stain_ch,
                CROP_SIZE, PATCH_SIZE,
                atlas, dseg_data, dseg_masked, target_mask,
                darr, output_dir, sub_id, dataset_name, local_patch_idx,
                stain_name=stain,
                spimquant_micr=spimquant_micr,
            )

            entry = {
                "subject_id": sub_id,
                "dataset": dataset_name,
                "region_group": group_name,
                "center_phys": [float(c) for c in center],
                "center_vox": [float(v) for v in center_vox],
                "stain_channel": stain_ch,
                "nii_path": str(nii_path),
                "qc_path": str(qc_path),
            }
            if seg_path is not None:
                entry["seg_path"] = str(seg_path)
            manifest_entries.append(entry)

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
