"""generate_roi_qc.py — atlas + mask QC figures for WT-normalized patches.

For every patch in ft_normalized/Abeta/ (found by walking meta.json files,
no manifest needed), renders a 4×3 figure:

  Row 0:  atlas ortho slices (axial, coronal, sagittal) zoomed on the patch
          center with brain-region contours and center marker
  Row 1:  raw 128³ crop slices (z = mid-10, mid, mid+10)
  Row 2:  raw + otsu2 mask overlay
  Row 3:  raw + otsu3 mask overlay

Saved as ``{patch_id}_roi_qc.png`` alongside the existing ``_qc.png``.

Usage:
    pixi run python scripts/generate_roi_qc.py             # all patches
    pixi run python scripts/generate_roi_qc.py --force      # overwrite existing
    pixi run python scripts/generate_roi_qc.py --dataset mouse_app_lecanemab_batch2
    pixi run python scripts/generate_roi_qc.py --patch sub-AS40F2_patch01
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from zarrnii import ZarrNiiAtlas

from lumivox.data.manifest import PREFERRED_SPIMQUANT

REPO_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = REPO_ROOT / "ft_normalized" / "Abeta"
LS_ROOT = Path("/nfs/trident3/lightsheet/prado")
LABELS_REF = REPO_ROOT / "lumivox" / "data" / "reference" / "seg-roi22_dseg.tsv"

# Dataset name overrides (manifest name -> on-disk directory name)
DS_DIR_OVERRIDES = {"mouse_app_vaccine_batch": "mouse_app_vaccine_batch1"}

# Target region IDs for contour overlay (cortex, hippocampus, striatum, cerebellum)
TARGET_REGION_IDS = {3, 4, 5, 6, 9, 10, 21, 22}

# Atlas zoom pads for the three columns
ATLAS_PADS = [None, 30, 15]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ds_root(dataset_name: str) -> Path:
    real = DS_DIR_OVERRIDES.get(dataset_name, dataset_name)
    return LS_ROOT / real


def _find_dseg(dataset_name: str, subject_id: str) -> Optional[Path]:
    ds = _ds_root(dataset_name)
    spq_dir = None
    for preferred in PREFERRED_SPIMQUANT.get(dataset_name, []):
        c = ds / "derivatives" / preferred
        if c.is_dir():
            spq_dir = c
            break
    if spq_dir is None:
        for c in sorted(ds.glob("derivatives/spimquant*")):
            if c.is_dir():
                spq_dir = c
                break
    if spq_dir is None:
        return None
    micr = spq_dir / subject_id / "micr"
    cands = sorted(micr.glob("*_seg-roi22_from-ABAv3_*_desc-deform_dseg.nii.gz"))
    return cands[0] if cands else None


def _find_labels(dataset_name: str, subject_id: str) -> Path:
    ds = _ds_root(dataset_name)
    # Try per-subject
    for c in sorted(ds.rglob(f"*{subject_id}*_seg-roi22_*_dseg.tsv")):
        return c
    # Try any tsv under spimquant
    for c in sorted(ds.rglob("*_seg-roi22_*_dseg.tsv")):
        return c
    # Bundled reference
    return LABELS_REF


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _pclip(arr: np.ndarray, lo: float = 5.0, hi: float = 99.5) -> Tuple[float, float]:
    nz = arr[arr > 0] if (arr > 0).any() else arr.ravel()
    if nz.size == 0:
        return 0.0, 1.0
    v0, v1 = float(np.percentile(nz, lo)), float(np.percentile(nz, hi))
    return (v0, v1) if v1 > v0 else (v0, v0 + 1.0)


def render_roi_qc(
    meta: Dict,
    patch_dir: Path,
    patch_id: str,
    dseg_data: np.ndarray,
    dseg_masked: np.ndarray,
    target_mask: np.ndarray,
    dseg_affine_inv,
    label_str: str,
    out_path: Path,
):
    """Render a 4×3 roi_qc figure and save it."""
    # Load patch NIfTIs
    raw = nib.load(str(patch_dir / f"{patch_id}_crop128.nii.gz")).get_fdata()
    o2 = nib.load(str(patch_dir / f"{patch_id}_seg_otsu2.nii.gz")).get_fdata().astype(np.uint8)
    o3 = nib.load(str(patch_dir / f"{patch_id}_seg_otsu3.nii.gz")).get_fdata().astype(np.uint8)

    # Atlas voxel coords
    center_phys = np.array(meta["center_phys"])
    vox_dseg = dseg_affine_inv @ center_phys
    di, dj, dk = (int(round(v)) for v in vox_dseg[:3])

    # Crop slice indices
    mid = raw.shape[0] // 2
    slices = [max(0, mid - 10), mid, min(raw.shape[0] - 1, mid + 10)]
    raw_lo, raw_hi = _pclip(raw)

    fig, axes = plt.subplots(4, 3, figsize=(15, 19))

    # --- Row 0: atlas ortho views (axial / coronal / sagittal) ---
    ortho_specs = [
        (dseg_masked[di], target_mask[di], dk, dj, f"axial  z={di}"),
        (dseg_masked[:, dj, :], target_mask[:, dj, :], dk, di, f"coronal  y={dj}"),
        (dseg_masked[:, :, dk], target_mask[:, :, dk], dj, di, f"sagittal  x={dk}"),
    ]
    for col, (sdata, smask, dot_x, dot_y, ttl) in enumerate(ortho_specs):
        ax = axes[0, col]
        ax.imshow(sdata, cmap="nipy_spectral", vmin=1, vmax=22, origin="lower")
        try:
            ax.contour(smask, colors="lime", linewidths=0.8)
        except ValueError:
            pass  # no contours if mask is all-zero
        ax.plot(dot_x, dot_y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        pad = 30
        ax.set_xlim(dot_x - pad, dot_x + pad)
        ax.set_ylim(dot_y - pad, dot_y + pad)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    # --- Row 1: raw crop slices ---
    for j, sl in enumerate(slices):
        ax = axes[1, j]
        ax.imshow(raw[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        ax.set_title(f"raw  z={sl}", fontsize=9)
        ax.axis("off")

    # --- Row 2: raw + otsu2 overlay ---
    n2 = int(meta.get("n_components_otsu2", 0))
    for j, sl in enumerate(slices):
        ax = axes[2, j]
        ax.imshow(raw[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        ov = np.ma.masked_where(o2[sl] == 0, o2[sl])
        ax.imshow(ov, cmap="autumn", alpha=0.55, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"otsu k=2 ({n2} comp)  z={sl}", fontsize=9)
        ax.axis("off")

    # --- Row 3: raw + otsu3 overlay ---
    n3 = int(meta.get("n_components_otsu3", 0))
    for j, sl in enumerate(slices):
        ax = axes[3, j]
        ax.imshow(raw[sl], cmap="gray", vmin=raw_lo, vmax=raw_hi, origin="lower")
        ov = np.ma.masked_where(o3[sl] == 0, o3[sl])
        ax.imshow(ov, cmap="cool", alpha=0.55, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"otsu k=3 ({n3} comp)  z={sl}", fontsize=9)
        ax.axis("off")

    fb = meta.get("wt_fallback_count", 0)
    fb_tag = f"  [WT global fallback]" if fb else ""
    fig.suptitle(
        f"{meta.get('dataset','')} / {meta.get('subject_id','')} / "
        f"{patch_id}  |  {label_str}\n"
        f"mu_wt={meta.get('mu_wt',0):.1f}  sigma_wt={meta.get('sigma_wt',0):.1f}  |  "
        f"k2={n2}  k3={n3}{fb_tag}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dataset", default=None)
    p.add_argument("--patch", default=None)
    args = p.parse_args()

    # Discover all patches from meta.json files
    all_metas: List[Tuple[Path, Dict]] = []
    for mp in sorted(FT_DIR.rglob("*_meta.json")):
        if "wt_references" in str(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        pid = mp.name.replace("_meta.json", "")
        if args.patch and pid != args.patch:
            continue
        if args.dataset and m.get("dataset") != args.dataset:
            continue
        all_metas.append((mp, m))

    if not all_metas:
        sys.exit("no patches found")

    print(f"generating roi_qc for {len(all_metas)} patches ...")

    # Group by (dataset, subject) for dseg caching
    by_subj: Dict[Tuple[str, str], List[Tuple[Path, Dict, str]]] = defaultdict(list)
    for mp, m in all_metas:
        pid = mp.name.replace("_meta.json", "")
        key = (m["dataset"], m["subject_id"])
        by_subj[key].append((mp, m, pid))

    n_done = 0
    n_skip = 0
    for (ds, sub), entries in sorted(by_subj.items()):
        # Load atlas once per subject
        dseg_path = _find_dseg(ds, sub)
        if dseg_path is None:
            print(f"  SKIP {ds}/{sub}: no dseg found")
            continue
        labels_path = _find_labels(ds, sub)

        atlas = ZarrNiiAtlas.from_files(dseg_path=str(dseg_path), labels_path=str(labels_path))
        dseg_data = atlas.dseg.darr.compute().squeeze()
        dseg_masked = np.ma.masked_where(dseg_data == 0, dseg_data)
        target_mask = np.isin(dseg_data, list(TARGET_REGION_IDS))
        dseg_affine_inv = atlas.dseg.affine.invert()

        for mp, m, pid in entries:
            out_path = mp.parent / f"{pid}_roi_qc.png"
            if out_path.exists() and not args.force:
                n_skip += 1
                continue

            # Look up label name at patch center
            center_phys = np.array(m["center_phys"])
            vox = dseg_affine_inv @ center_phys
            di = int(round(vox[0]))
            di = max(0, min(di, dseg_data.shape[0] - 1))
            dj = int(round(vox[1]))
            dj = max(0, min(dj, dseg_data.shape[1] - 1))
            dk = int(round(vox[2]))
            dk = max(0, min(dk, dseg_data.shape[2] - 1))
            lval = dseg_data[di, dj, dk]
            lnames = atlas.labels_df[atlas.labels_df["index"] == int(lval)]["name"].values
            label_str = lnames[0] if len(lnames) > 0 else f"label={int(lval)}"

            try:
                render_roi_qc(
                    m, mp.parent, pid,
                    dseg_data, dseg_masked, target_mask, dseg_affine_inv,
                    label_str, out_path,
                )
                n_done += 1
                print(f"  {ds}/{sub}/{pid}  ->  {out_path.name}  ({label_str})")
            except Exception as e:
                print(f"  FAIL {pid}: {e}")

    print(f"\ndone.  wrote={n_done}  skipped={n_skip}")


if __name__ == "__main__":
    main()
