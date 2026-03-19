"""Patch manifest builder for OME-Zarr datasets with region-masked sampling.

Discovers subjects across one or more SPIMquant dataset roots, samples patch
centers from specified brain regions using zarrnii's atlas tools, and saves
a manifest for lazy loading during training.

Usage:
    python -m lumivox.data.manifest \
        --dataset-roots /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2 \
        --stain Abeta \
        --regions L_Isocortex R_Isocortex "L_Hippocampal formation" "R_Hippocampal formation" \
        --n-patches 10000 \
        --output manifests/patches.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Fallback stain->channel map (only used if omero metadata is missing)
STAIN_CHANNEL_MAP_FALLBACK = {
    "IBA1": 0,
    "Abeta": 1,
    "CD31": 2,
}

# Preferred SPIMquant derivative directory per dataset.
# Keys are matched against the dataset root's directory name.
# Values are ordered by preference — the first existing match wins.
# If no match is found, falls back to globbing derivatives/spimquant*.
PREFERRED_SPIMQUANT = {
    "mouse_app_lecanemab_batch2": [
        "spimquant-v0.6.0rc2_84a605e_ozx",
        "spimquant_c270a40_atropos",
    ],
    "mouse_app_lecanemab_batch3": [
        "spimquant-v0.6.0rc2_84a605e_ozx",
    ],
    "mouse_app_lecanemab_ki3_batch1": [
        "spimquant_v0.5.0-alpha1",
    ],
    "mouse_app_lecanemab_ki3_batch2": [
        "spimquant_c270a40_atropos",
    ],
    "mouse_app_lecanemab_ki3_batch3": [
        "spimquant_c270a40_atropos",
    ],
    "mouse_app_vaccine_batch": [
        "spimquant_c270a40_atropos",
    ],
}


def resolve_stain_channel(zarr_path: str, stain: str, fullres_zarr_path: Optional[str] = None) -> int:
    """Look up the channel index for a stain from the OME-Zarr omero metadata.

    Resampled zarr files often lack omero metadata, so ``fullres_zarr_path``
    is checked as a fallback (the channel order is the same).
    Falls back to STAIN_CHANNEL_MAP_FALLBACK if no omero labels are found.
    """
    for path in [zarr_path, fullres_zarr_path]:
        if path is None:
            continue
        try:
            import zarr as zarr_lib
            store = zarr_lib.open(path, mode="r")
            omero = dict(store.attrs).get("omero", {})
            channels = omero.get("channels", [])
            for i, ch in enumerate(channels):
                if ch.get("label", "").lower() == stain.lower():
                    return i
        except Exception:
            continue
    return STAIN_CHANNEL_MAP_FALLBACK.get(stain, 0)


def _prefer_standard_acq(paths: List[Path]) -> List[Path]:
    """Filter zarr paths to prefer standard acq-imaris4x over angled variants.

    Excludes 45deg/90deg acquisitions when a standard one is available.
    If only angled ones exist, returns them as-is.
    """
    standard = [p for p in paths if "45deg" not in p.name and "90deg" not in p.name]
    return standard if standard else paths


def discover_subjects(
    dataset_root: str,
    seg_level: str = "roi22",
) -> List[Dict[str, str]]:
    """Find subjects that have both OME-Zarr data and segmentation masks.

    Prefers the resampled 4 um OME-Zarr if available, otherwise falls back
    to the full-resolution OME-Zarr in ``bids/sub-*/micr/``.

    Returns list of dicts with keys:
        subject_id, zarr_path, zarr_source ('resampled' or 'fullres'),
        dseg_path, labels_path
    """
    root = Path(dataset_root)
    bids_dir = root / "bids"
    resampled_dir = root / "bids" / "derivatives" / "resampled"

    # Resolve the SPIMquant derivatives directory.
    # Check PREFERRED_SPIMQUANT first (keyed by dataset dir name),
    # then fall back to globbing derivatives/spimquant*.
    dataset_name = root.name
    spimquant_dir = None

    for preferred in PREFERRED_SPIMQUANT.get(dataset_name, []):
        candidate = root / "derivatives" / preferred
        if candidate.is_dir():
            spimquant_dir = candidate
            break

    if spimquant_dir is None:
        spimquant_candidates = sorted(root.glob("derivatives/spimquant*"))
        spimquant_dir = spimquant_candidates[0] if spimquant_candidates else None

    subjects: List[Dict[str, str]] = []

    if spimquant_dir is None:
        print(f"  Warning: no spimquant* directory under {root / 'derivatives'}")
        return subjects

    if not bids_dir.exists():
        print(f"  Warning: no bids directory at {bids_dir}")
        return subjects

    # Collect all subject IDs from bids/
    sub_dirs = sorted(
        d for d in bids_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )

    for sub_dir in sub_dirs:
        subject_id = sub_dir.name

        # Always locate the full-res zarr (needed for omero channel metadata).
        # Prefer the standard acq-imaris4x acquisition; skip 45deg/90deg variants.
        fullres_micr = sub_dir / "micr"
        fullres_candidates = sorted(fullres_micr.glob("*_SPIM.ome.zarr"))
        fullres_candidates = _prefer_standard_acq(fullres_candidates)
        fullres_zarr = str(fullres_candidates[0]) if fullres_candidates else None

        # Try resampled 4um zarr first, then fall back to full-res
        zarr_path: Optional[str] = None
        zarr_source = "fullres"

        if resampled_dir.exists():
            resampled_micr = resampled_dir / subject_id / "micr"
            res4um = sorted(resampled_micr.glob("*_res-4um_SPIM.ome.zarr"))
            res4um = _prefer_standard_acq(res4um)
            if res4um:
                zarr_path = str(res4um[0])
                zarr_source = "resampled"

        if zarr_path is None:
            if fullres_zarr is not None:
                zarr_path = fullres_zarr
                zarr_source = "fullres"

        if zarr_path is None:
            continue

        # Deformed segmentation mask (NIfTI)
        spimquant_micr = spimquant_dir / subject_id / "micr"
        dseg_candidates = sorted(spimquant_micr.glob(
            f"*_seg-{seg_level}_from-ABAv3_*_desc-deform_dseg.nii.gz"
        ))
        if not dseg_candidates:
            continue
        dseg_path = str(dseg_candidates[0])

        # Labels TSV (any stain variant -- label IDs are identical).
        # Some spimquant runs don't produce per-subject TSVs, so fall back
        # to any TSV in the spimquant dir, then to the bundled reference.
        tsv_candidates = sorted(spimquant_micr.glob(
            f"*_seg-{seg_level}_from-ABAv3_*_desc-deform_dseg.tsv"
        ))
        if not tsv_candidates:
            tsv_candidates = sorted(spimquant_dir.rglob(
                f"*_seg-{seg_level}_*_dseg.tsv"
            ))
        if tsv_candidates:
            labels_path = str(tsv_candidates[0])
        else:
            # Use bundled reference TSV (atlas labels are fixed across datasets)
            ref = Path(__file__).parent / "reference" / f"seg-{seg_level}_dseg.tsv"
            if not ref.exists():
                continue
            labels_path = str(ref)

        subjects.append({
            "subject_id": subject_id,
            "dataset_root": dataset_root,
            "zarr_path": zarr_path,
            "zarr_source": zarr_source,
            "fullres_zarr": fullres_zarr,
            "dseg_path": dseg_path,
            "labels_path": labels_path,
        })

    return subjects


def build_patch_manifest(
    dataset_roots: Sequence[str],
    stain: str = "Abeta",
    regions: Optional[Sequence[str]] = None,
    region_groups: Optional[Dict[str, tuple]] = None,
    n_patches: int = 10000,
    patch_size: int = 256,
    crop_size: int = 96,
    seg_level: str = "roi22",
    seed: int = 42,
) -> Dict:
    """Build a patch manifest by sampling centers from brain regions.

    Discovers subjects across all dataset_roots, loads per-subject
    segmentation atlases with zarrnii, samples patch centers from the
    specified brain regions in physical coordinates, and returns a
    manifest dict ready for JSON serialization.

    Regions can be specified in two ways:
      - ``regions``: flat list of region names, sampled uniformly.
      - ``region_groups``: dict mapping group name to
        ``(region_names_list, weight)`` for weighted sampling.
        Weights are normalised to sum to 1.

    Centers are in physical space (mm) so they can be used to crop from
    any resolution level of the OME-Zarr volume.
    """
    from zarrnii import ZarrNiiAtlas

    rng = np.random.default_rng(seed)

    # Resolve region specification
    if region_groups is not None:
        # Weighted sampling: compute n_patches per group
        total_weight = sum(w for _, w in region_groups.values())
        group_patches = {}
        allocated = 0
        group_names = list(region_groups.keys())
        for i, name in enumerate(group_names):
            region_list, weight = region_groups[name]
            if i == len(group_names) - 1:
                n = n_patches - allocated  # last group gets remainder
            else:
                n = int(round(n_patches * weight / total_weight))
            group_patches[name] = (region_list, n)
            allocated += n
        all_region_names = []
        for region_list, _ in region_groups.values():
            all_region_names.extend(region_list)
        print(f"Region groups:")
        for name, (rlist, n) in group_patches.items():
            print(f"  {name}: {n} patches from {rlist}")
    elif regions is not None:
        group_patches = {"all": (list(regions), n_patches)}
        all_region_names = list(regions)
    else:
        raise ValueError("Must specify either regions or region_groups")

    # Discover subjects across all dataset roots
    all_subjects: List[Dict[str, str]] = []
    for root in dataset_roots:
        subs = discover_subjects(root, seg_level=seg_level)
        all_subjects.extend(subs)
        print(f"  {root}: {len(subs)} subjects")

    if not all_subjects:
        raise RuntimeError(
            f"No valid subjects found across {dataset_roots}. "
            "Each subject needs a resampled 4um OME-Zarr and a "
            f"deformed {seg_level} segmentation in derivatives/spimquant/."
        )

    n_subjects = len(all_subjects)
    print(f"Total subjects: {n_subjects}")

    # Sample patch centers per subject
    patches: List[Dict] = []
    for subj in all_subjects:
        atlas = ZarrNiiAtlas.from_files(
            dseg_path=subj["dseg_path"],
            labels_path=subj["labels_path"],
        )

        # Sample from each region group separately, distribute evenly per subject
        centers = []
        for group_name, (region_list, group_n) in group_patches.items():
            n_this_subj = group_n // n_subjects
            # Distribute remainder across first subjects
            subj_idx = all_subjects.index(subj)
            if subj_idx < (group_n % n_subjects):
                n_this_subj += 1
            if n_this_subj == 0:
                continue
            subj_seed = int(rng.integers(0, 2**31))
            grp_centers = atlas.sample_region_patches(
                n_patches=n_this_subj,
                region_ids=region_list,
                seed=subj_seed,
            )
            centers.extend(grp_centers)

        if not centers:
            continue

        # Resolve stain channel from omero metadata
        stain_ch = resolve_stain_channel(
            subj["zarr_path"], stain, subj.get("fullres_zarr"),
        )

        # Convert physical centers → resampled voxel coordinates.
        # Physical → full-res voxel (via full-res zarr affine, which is
        # the only reliable coordinate system), then full-res voxel →
        # resampled voxel (via JSON sidecar scale factors).
        fullres_zarr = subj.get("fullres_zarr")
        if fullres_zarr and subj["zarr_source"] == "resampled":
            from zarrnii import ZarrNii as _ZN

            fullres_inv = _ZN.from_ome_zarr(
                fullres_zarr, channels=[0],
            ).affine.invert()

            # Read JSON sidecar for resampled → fullres scale
            sidecar_path = Path(subj["zarr_path"] + ".json")
            if sidecar_path.exists():
                with open(sidecar_path) as _f:
                    sidecar = json.load(_f)
                sc = sidecar["fullres_to_resampled"]["scale"]
                res_scale = np.array([sc["z"], sc["y"], sc["x"]])
            else:
                # Fallback: compute from voxel sizes
                res_scale = np.array([0.6875, 0.40625, 0.40625])

            for center in centers:
                vox_fullres = fullres_inv @ np.array(center)
                vox_resampled = vox_fullres * res_scale
                patches.append({
                    "subject_id": subj["subject_id"],
                    "dataset_root": subj["dataset_root"],
                    "zarr_path": subj["zarr_path"],
                    "zarr_source": subj["zarr_source"],
                    "stain_channel": stain_ch,
                    "center_phys": [float(c) for c in center],
                    "center_vox": [float(v) for v in vox_resampled],
                })
        else:
            # Full-res path: crop_centered works, store physical only
            for center in centers:
                patches.append({
                    "subject_id": subj["subject_id"],
                    "dataset_root": subj["dataset_root"],
                    "zarr_path": subj["zarr_path"],
                    "zarr_source": subj["zarr_source"],
                    "stain_channel": stain_ch,
                    "center_phys": [float(c) for c in center],
                })

        print(f"  {subj['subject_id']}: {len(centers)} patches (stain={stain} -> ch {stain_ch})")

    # Shuffle so subjects are interleaved during training
    rng.shuffle(patches)

    # Store region config
    if region_groups is not None:
        regions_config = {name: {"regions": rlist, "weight": w}
                          for name, (rlist, w) in region_groups.items()}
    else:
        regions_config = {"all": {"regions": all_region_names, "weight": 1.0}}

    manifest = {
        "config": {
            "dataset_roots": list(dataset_roots),
            "stain": stain,
            "region_groups": regions_config,
            "n_patches": len(patches),
            "patch_size": patch_size,
            "crop_size": crop_size,
            "seg_level": seg_level,
            "seed": seed,
        },
        "patches": patches,
    }

    return manifest


def save_manifest(manifest: Dict, path: str) -> None:
    """Write manifest to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    n = manifest["config"]["n_patches"]
    print(f"Manifest saved: {out} ({n} patches)")


def load_manifest(path: str) -> Dict:
    """Read manifest from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build patch manifest for OME-Zarr SSL pretraining",
    )
    parser.add_argument(
        "--dataset-roots", nargs="+", required=True,
        help="One or more SPIMquant dataset root directories",
    )
    parser.add_argument(
        "--stain", default="Abeta", choices=list(STAIN_CHANNEL_MAP_FALLBACK),
    )
    parser.add_argument(
        "--regions", nargs="+",
        default=[
            "L_Isocortex", "R_Isocortex",
            "L_Hippocampal formation", "R_Hippocampal formation",
        ],
    )
    parser.add_argument("--n-patches", type=int, default=10000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--seg-level", default="roi22")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./manifests/patches.json")
    args = parser.parse_args()

    manifest = build_patch_manifest(
        dataset_roots=args.dataset_roots,
        stain=args.stain,
        regions=args.regions,
        n_patches=args.n_patches,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        seg_level=args.seg_level,
        seed=args.seed,
    )
    save_manifest(manifest, args.output)


if __name__ == "__main__":
    main()
