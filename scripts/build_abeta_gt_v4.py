"""Build the v4 Abeta fine-tuning ground-truth patch set.

Same approach as v3 but scaled to 250 patches and with ki3_batch3 dropped
(which also drops the WT sub-C57BL6, whose mask was unusable).

Two sampling tracks:

  (A) OZX track  (150 patches): atlas-sampled across 3 datasets
      (batch2 / batch3 / vaccine_batch1), stratified 40/30/20/10 across
      cortex/hippocampus/striatum/cerebellum. For each patch the mask is
      derived from the SPIMquant level-0 .ozx file (downsampled to 128^3
      with block-max pooling to preserve sparse plaques).

  (B) v0.5 patches track  (100 patches, 2x weight per the spec): for
      ki3_batch1, the v0.5 SPIMquant variant already produced 256^3
      patches at full-res voxel size with matching th900 binary masks.
      We pull 100 of those at random across all subjects/labels. fullres
      = the patch as-is, 4um = 256^3 resampled to 128^3 (voxels become
      ~3.25 x 3.25 x 4.4 um, close to 4 um iso per the spec).

Per-patch output: crop_4um.nii.gz, crop_fullres.nii.gz, mask_4um.nii.gz,
mask_fullres.nii.gz.

QC PNGs (3-panel, mid-slice): 4um axial, fullres axial, fullres + mask
overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import zarr
from scipy.ndimage import zoom as ndi_zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumivox.data.manifest import resolve_stain_channel
from zarrnii import ZarrNii, ZarrNiiAtlas


PATCH_4UM = 128
RES_4UM_MM = 0.004
PATCH_PHYS_UM = PATCH_4UM * RES_4UM_MM * 1000.0  # 512 um

REGION_GROUPS = {
    "cortex":      (["L_Isocortex",             "R_Isocortex"],             0.40),
    "hippocampus": (["L_Hippocampal formation", "R_Hippocampal formation"], 0.30),
    "striatum":    (["L_Striatum",              "R_Striatum"],              0.20),
    "cerebellum":  (["L_Cerebellum",            "R_Cerebellum"],            0.10),
}

# Each OZX dataset config:
#   root            = on-disk dataset root
#   spimquant       = name of the spimquant variant subdir
#   ozx_subdir      = where the OZX lives within each subject ("micr" or "seg")
#   seg_level       = atlas segmentation level — "roi22" for older variants,
#                     "coarse" for v0.7.0rc3 (both give the 22-region split)
#   dseg_subdirs    = list of per-subject subdirs to search for dseg nii.gz
#                     (older variants put it in micr/; v0.7.0rc3 puts it in parc/)
#   dseg_glob_extra = optional extra glob qualifier (e.g. "desc-deform_" for
#                     older variants). v0.7.0rc3 drops the desc-deform marker.
#   tpl_subdir      = subdir under spimquant root that holds the labels TSV
#                     template for v0.7.0rc3-style outputs ("tpl-ABAv3")
DATASETS_OZX = [
    {
        "root":      "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2",
        "spimquant": "spimquant-v0.6.0rc2_84a605e_ozx",
        "ozx_subdir": "micr",
        "seg_level": "roi22",
        "dseg_subdirs": ["micr"],
        "dseg_glob_extra": "desc-deform_",
        "tpl_subdir": None,
    },
    {
        "root":      "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3",
        "spimquant": "spimquant-v0.6.0rc2_84a605e_ozx",
        "ozx_subdir": "micr",
        "seg_level": "roi22",
        "dseg_subdirs": ["micr"],
        "dseg_glob_extra": "desc-deform_",
        "tpl_subdir": None,
    },
    # ki3_batch3 dropped for v4: per-subject mask densities on cortex (7-29%)
    # were 5-10x higher than batch2/3/vaccine (<5%), suggesting the v0.7.0rc3
    # otsu+k3i2 threshold for these subjects is calibrated less stringently and
    # produces over-segmented masks. Also drops sub-C57BL6 (WT mouse with a
    # 76%-positive mask — no real Abeta to threshold).
    {
        "root":      "/nfs/trident3/lightsheet/prado/mouse_app_vaccine_batch1",
        "spimquant": "spimquant_v0.6.0rc1",
        "ozx_subdir": "micr",
        "seg_level": "roi22",
        "dseg_subdirs": ["micr"],
        "dseg_glob_extra": "desc-deform_",
        "tpl_subdir": None,
    },
]

V05_CFG = {
    "root":      "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "spimquant": "spimquant_v0.5.0-alpha1",
    # Templates evaluate {subj} = "sub-XXX" (with the sub- prefix).
    "raw_dir":   "{subj}_sample-brain_acq-imaris4x_stain-Abeta_seg-roi22_from-ABAv3_level-0_desc-raw_SPIM.patches",
    "mask_dir":  "{subj}_sample-brain_acq-imaris4x_stain-Abeta_seg-roi22_from-ABAv3_level-0_desc-th900_mask.patches",
}


# ---------------------------------------------------------------------------
# OZX-track discovery (custom because we override spimquant per dataset and
# the OZX may live under either micr/ or seg/)
# ---------------------------------------------------------------------------

def _prefer_standard_acq(paths: List[Path]) -> List[Path]:
    std = [p for p in paths if "45deg" not in p.name and "90deg" not in p.name]
    return std if std else paths


def discover_ozx_subjects(cfg: Dict) -> List[Dict]:
    """Return per-subject bundle: zarr_path, fullres_zarr, dseg_path, labels_path, ozx_path."""
    root = Path(cfg["root"])
    bids_dir = root / "bids"
    resampled_dir = bids_dir / "derivatives" / "resampled"
    spimquant_dir = root / "derivatives" / cfg["spimquant"]
    seg_level = cfg["seg_level"]
    dseg_glob_extra = cfg.get("dseg_glob_extra", "")
    if not spimquant_dir.is_dir():
        print(f"  Warning: {spimquant_dir} not a dir")
        return []
    if not bids_dir.exists():
        print(f"  Warning: no bids at {bids_dir}")
        return []

    # Bundled labels TSV fallback for the roi22 layout (atlas labels are
    # dataset-independent for older variants).
    ref_tsv = (
        Path(__file__).resolve().parent.parent
        / "lumivox" / "data" / "reference" / f"seg-{seg_level}_dseg.tsv"
    )
    # Per-variant tpl-ABAv3 TSV (the v0.7.0rc3 layout doesn't drop a per-subject
    # labels TSV, so the labels live at <spimquant>/<tpl_subdir>/...)
    tpl_tsv = None
    if cfg.get("tpl_subdir"):
        tpl_path = spimquant_dir / cfg["tpl_subdir"] / f"tpl-ABAv3_seg-{seg_level}_dseg.tsv"
        if tpl_path.exists():
            tpl_tsv = str(tpl_path)

    subjects = []
    for sub_dir in sorted(bids_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        subject_id = sub_dir.name

        # Fullres zarr (always need it for omero channel info + as fullres source)
        fr_micr = sub_dir / "micr"
        fr_cands = _prefer_standard_acq(sorted(fr_micr.glob("*_SPIM.ome.zarr")))
        fullres_zarr = str(fr_cands[0]) if fr_cands else None

        # Resampled (4um) zarr
        zarr_path = None
        zarr_source = "fullres"
        rs_micr = resampled_dir / subject_id / "micr"
        if rs_micr.exists():
            rs_cands = _prefer_standard_acq(sorted(rs_micr.glob("*_res-4um_SPIM.ome.zarr")))
            if rs_cands:
                zarr_path = str(rs_cands[0])
                zarr_source = "resampled"
        if zarr_path is None:
            zarr_path = fullres_zarr

        if zarr_path is None or fullres_zarr is None:
            continue

        # Sidecar (resampled->fullres scale)
        sidecar = Path(zarr_path + ".json")
        if zarr_source == "resampled" and not sidecar.exists():
            continue

        # dseg (deformed Allen atlas labels). The seg_level + dseg subdir layout
        # comes from cfg so this script handles both older (seg-roi22 in micr/
        # with desc-deform) and v0.7.0rc3 (seg-coarse in parc/, no desc-deform).
        dseg_paths = []
        for sd in cfg["dseg_subdirs"]:
            d = spimquant_dir / subject_id / sd
            if d.exists():
                dseg_paths.extend(sorted(d.glob(
                    f"*_seg-{seg_level}_from-ABAv3_*{dseg_glob_extra}dseg.nii.gz"
                )))
        if not dseg_paths:
            continue
        dseg_path = str(dseg_paths[0])

        # labels TSV
        labels_path = None
        if tpl_tsv:
            labels_path = tpl_tsv
        else:
            tsv_cands = []
            for sd in cfg["dseg_subdirs"] + ["micr", "seg"]:
                d = spimquant_dir / subject_id / sd
                if d.exists():
                    tsv_cands.extend(sorted(d.glob(
                        f"*_seg-{seg_level}_from-ABAv3_*{dseg_glob_extra}dseg.tsv"
                    )))
            if not tsv_cands:
                tsv_cands = sorted(spimquant_dir.rglob(f"*_seg-{seg_level}_*_dseg.tsv"))
            if tsv_cands:
                labels_path = str(tsv_cands[0])
            elif ref_tsv.exists():
                labels_path = str(ref_tsv)
        if labels_path is None:
            continue

        # OZX mask (Abeta, level-0)
        ozx_search = spimquant_dir / subject_id / cfg["ozx_subdir"]
        ozx_cands = sorted(ozx_search.glob("*stain-Abeta_level-0*mask.ozx"))
        if not ozx_cands:
            continue
        ozx_path = str(ozx_cands[0])

        subjects.append({
            "subject_id": subject_id,
            "dataset_root": cfg["root"],
            "dataset_name": Path(cfg["root"]).name,
            "spimquant_subdir": cfg["spimquant"],
            "zarr_path": zarr_path,
            "zarr_source": zarr_source,
            "fullres_zarr": fullres_zarr,
            "sidecar": str(sidecar),
            "dseg_path": dseg_path,
            "labels_path": labels_path,
            "ozx_path": ozx_path,
        })
    return subjects


# ---------------------------------------------------------------------------
# OZX-track sampling (atlas-driven, stratified across ROIs)
# ---------------------------------------------------------------------------

def sample_ozx_patches(
    all_subjects: List[Dict],
    region_groups: Dict,
    n_patches: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)

    # Per-group targets across the OZX track
    targets: Dict[str, int] = {}
    allocated = 0
    names = list(region_groups.keys())
    total_w = sum(w for _, w in region_groups.values())
    for i, name in enumerate(names):
        _, w = region_groups[name]
        if i == len(names) - 1:
            targets[name] = n_patches - allocated
        else:
            n = int(round(n_patches * w / total_w))
            targets[name] = n
            allocated += n
    print(f"OZX per-group targets: {targets}  (sum={sum(targets.values())})")

    n_subj = len(all_subjects)
    if n_subj == 0:
        return []

    patches = []
    for subj_idx, subj in enumerate(all_subjects):
        atlas = ZarrNiiAtlas.from_files(
            dseg_path=subj["dseg_path"],
            labels_path=subj["labels_path"],
        )
        centers_with_group = []
        for grp_name, tgt in targets.items():
            n_this = tgt // n_subj + (1 if subj_idx < (tgt % n_subj) else 0)
            if n_this == 0:
                continue
            grp_seed = int(rng.integers(0, 2**31))
            try:
                cs = atlas.sample_region_patches(
                    n_patches=n_this,
                    region_ids=region_groups[grp_name][0],
                    seed=grp_seed,
                )
            except Exception as exc:
                print(f"    {subj['subject_id']} {grp_name}: sample failed ({exc})")
                continue
            for c in cs:
                centers_with_group.append((c, grp_name))
        if not centers_with_group:
            continue

        stain_ch = resolve_stain_channel(
            subj["zarr_path"], "Abeta", subj["fullres_zarr"]
        )

        # Resampled <-> fullres scale (sidecar)
        sidecar = Path(subj["sidecar"])
        sc = json.loads(sidecar.read_text())["fullres_to_resampled"]["scale"]
        res_scale = np.array([sc["z"], sc["y"], sc["x"]])

        fr_inv = ZarrNii.from_ome_zarr(subj["fullres_zarr"], channels=[0]).affine.invert()

        for c, grp in centers_with_group:
            vox_fr = fr_inv @ np.array(c)
            vox_res = vox_fr * res_scale
            patches.append({
                **{k: subj[k] for k in (
                    "subject_id", "dataset_root", "dataset_name",
                    "spimquant_subdir", "zarr_path", "fullres_zarr",
                    "sidecar", "ozx_path"
                )},
                "stain_channel": stain_ch,
                "region_group": grp,
                "center_phys": [float(v) for v in c],
                "center_vox": [float(v) for v in vox_res],
                "center_vox_fullres": [float(v) for v in vox_fr],
                "fullres_to_resampled_scale": {"z": sc["z"], "y": sc["y"], "x": sc["x"]},
            })
        print(f"  {subj['subject_id']}: kept {len(centers_with_group)} OZX patches")

    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(patches)
    return patches


# ---------------------------------------------------------------------------
# OZX-track extraction (crops + mask)
# ---------------------------------------------------------------------------

def open_zarr_array(zarr_path: Path):
    """Open level-0 of an OME-Zarr (handles v0.4 '0' and other layouts)."""
    p0 = zarr_path / "0"
    if p0.exists() and (p0 / ".zarray").exists():
        return zarr.open(str(p0), mode="r")
    p1 = zarr_path / "s0"
    if p1.exists():
        return zarr.open(str(p1), mode="r")
    return zarr.open(str(zarr_path), mode="r")[0]


def fullres_voxel_um(fullres_zarr: Path) -> List[float]:
    zattrs_path = fullres_zarr / ".zattrs"
    if zattrs_path.exists():
        meta = json.loads(zattrs_path.read_text())
        scale_mm = meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        return [scale_mm[1] * 1000.0, scale_mm[2] * 1000.0, scale_mm[3] * 1000.0]
    raise RuntimeError(f"no .zattrs at {fullres_zarr}")


def extract_ozx_box(
    mask_darr,
    center_fr: np.ndarray,
    half_fr: np.ndarray,
) -> np.ndarray:
    z0 = int(round(center_fr[0] - half_fr[0])); z1 = int(round(center_fr[0] + half_fr[0]))
    y0 = int(round(center_fr[1] - half_fr[1])); y1 = int(round(center_fr[1] + half_fr[1]))
    x0 = int(round(center_fr[2] - half_fr[2])); x1 = int(round(center_fr[2] + half_fr[2]))
    Z, Y, X = mask_darr.shape
    sz0, sz1 = max(0, z0), min(Z, z1)
    sy0, sy1 = max(0, y0), min(Y, y1)
    sx0, sx1 = max(0, x0), min(X, x1)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        slab = np.asarray(mask_darr[sz0:sz1, sy0:sy1, sx0:sx1].compute())
        out[sz0 - z0:sz0 - z0 + slab.shape[0],
            sy0 - y0:sy0 - y0 + slab.shape[1],
            sx0 - x0:sx0 - x0 + slab.shape[2]] = (slab > 0).astype(np.uint8)
    return out


def downsample_mask_to_128(box: np.ndarray) -> np.ndarray:
    """Block-max pool an arbitrary-shape binary box to 128^3 (preserves sparse positives)."""
    out = np.zeros((PATCH_4UM, PATCH_4UM, PATCH_4UM), dtype=np.uint8)
    nz = np.argwhere(box > 0)
    if nz.size:
        sz = PATCH_4UM / box.shape[0]
        sy = PATCH_4UM / box.shape[1]
        sx = PATCH_4UM / box.shape[2]
        iz = np.minimum((nz[:, 0] * sz).astype(np.int64), PATCH_4UM - 1)
        iy = np.minimum((nz[:, 1] * sy).astype(np.int64), PATCH_4UM - 1)
        ix = np.minimum((nz[:, 2] * sx).astype(np.int64), PATCH_4UM - 1)
        out[iz, iy, ix] = 1
    return out


def extract_ozx_record(entry: Dict) -> Dict:
    """Extract crop_4um (128^3) + crop_fullres (anisotropic) + mask at both resolutions."""
    cv = entry["center_vox"]
    z0 = int(round(cv[0] - PATCH_4UM / 2)); z1 = z0 + PATCH_4UM
    y0 = int(round(cv[1] - PATCH_4UM / 2)); y1 = y0 + PATCH_4UM
    x0 = int(round(cv[2] - PATCH_4UM / 2)); x1 = x0 + PATCH_4UM

    r4_arr = open_zarr_array(Path(entry["zarr_path"]))
    stain_ch = entry["stain_channel"]
    _, Z4, Y4, X4 = r4_arr.shape
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z4 or y1 > Y4 or x1 > X4:
        return {"skip_reason": f"4um OOB ({(z0,y0,x0)}->{(z1,y1,x1)} vs {(Z4,Y4,X4)})"}
    crop_4um = np.asarray(r4_arr[stain_ch, z0:z1, y0:y1, x0:x1], dtype=np.uint16)

    fr_zarr = Path(entry["fullres_zarr"])
    fr_um = fullres_voxel_um(fr_zarr)
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
        return {"skip_reason": f"fullres OOB ({(fz0,fy0,fx0)}->{(fz1,fy1,fx1)} vs {(FZ,FY,FX)})"}
    crop_fr = np.asarray(fr_arr[stain_ch, fz0:fz1, fy0:fy1, fx0:fx1], dtype=np.uint16)

    # Mask from OZX (same voxel grid as fullres zarr)
    znimg_mask = ZarrNii.from_ome_zarr(entry["ozx_path"], channels=[0])
    mask_darr = znimg_mask.darr[0]
    if tuple(mask_darr.shape) != (FZ, FY, FX):
        return {"skip_reason": f"OZX shape {tuple(mask_darr.shape)} != fullres {(FZ,FY,FX)}"}
    mask_fr = extract_ozx_box(
        mask_darr,
        np.array(cv_fr, dtype=np.float64),
        np.array([pz / 2.0, py / 2.0, px / 2.0]),
    )
    # The box returned has shape (pz, py, px) — same as crop_fr by construction.
    mask_4um = downsample_mask_to_128(mask_fr)

    return {
        "crop_4um": crop_4um,
        "crop_fullres": crop_fr,
        "mask_4um": mask_4um,
        "mask_fullres": mask_fr,
        "fullres_voxel_um": fr_um,
        "box_4um_zyx": [z0, y0, x0, z1, y1, x1],
        "box_fr_zyx": [fz0, fy0, fx0, fz1, fy1, fx1],
        "fr_patch_shape": [pz, py, px],
    }


# ---------------------------------------------------------------------------
# v0.5 track: enumerate + sample + extract
# ---------------------------------------------------------------------------

V05_NAME_RE = re.compile(
    r"^(?P<subj>sub-[^_]+)_seg-roi22_label-(?P<label>[A-Z0-9_]+)_patch-(?P<idx>\d+)_(SPIM|mask)\.nii(?:\.gz)?$"
)


def enumerate_v05_patches(cfg: Dict) -> List[Dict]:
    root = Path(cfg["root"]) / "derivatives" / cfg["spimquant"]
    items = []
    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        subj = sub_dir.name
        raw_dir = sub_dir / "micr" / cfg["raw_dir"].format(subj=subj)
        mask_dir = sub_dir / "micr" / cfg["mask_dir"].format(subj=subj)
        if not raw_dir.exists() or not mask_dir.exists():
            continue
        for raw_path in sorted(raw_dir.glob("*_SPIM.nii*")):
            m = V05_NAME_RE.match(raw_path.name)
            if not m:
                continue
            mask_name = raw_path.name.rsplit("_SPIM.", 1)[0] + "_mask.nii.gz"
            mask_path = mask_dir / mask_name
            if not mask_path.exists():
                # try .nii without gz
                mp2 = mask_dir / (raw_path.name.rsplit("_SPIM.", 1)[0] + "_mask.nii")
                if mp2.exists():
                    mask_path = mp2
                else:
                    continue
            items.append({
                "subject_id": subj,
                "label": m.group("label"),
                "patch_idx": int(m.group("idx")),
                "raw_path": str(raw_path),
                "mask_path": str(mask_path),
                "dataset_root": cfg["root"],
                "dataset_name": Path(cfg["root"]).name,
                "spimquant_subdir": cfg["spimquant"],
            })
    return items


def sample_v05_patches(all_items: List[Dict], n: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    if len(all_items) <= n:
        return list(all_items)
    idxs = rng.choice(len(all_items), size=n, replace=False)
    return [all_items[i] for i in idxs]


def _to_zyx(arr: np.ndarray, nii_img: nib.Nifti1Image) -> np.ndarray:
    """Reorder a nibabel data array (typically XYZ) to ZYX zarr-convention.

    nibabel returns the array in the order described by the affine; for these
    files the affine is diagonal so this is equivalent to a transpose.
    """
    # nibabel canonical: array is indexed [i, j, k] where (i, j, k) map to
    # whatever the affine encodes. For diagonal-positive affines that's XYZ.
    # We want ZYX for consistent saving with the OZX track's nibabel writes
    # (which also use diagonal affines but in (z,y,x) order).
    return np.transpose(arr, (2, 1, 0))


def extract_v05_record(item: Dict) -> Dict:
    """For a v0.5 patch: load raw + mask, then resample to 128^3 for the 4um pair."""
    raw_nii = nib.load(item["raw_path"])
    mask_nii = nib.load(item["mask_path"])
    raw = np.asarray(raw_nii.get_fdata(), dtype=np.float32)
    mask = (np.asarray(mask_nii.get_fdata()) > 0).astype(np.uint8)
    raw_zyx = _to_zyx(raw, raw_nii)
    mask_zyx = _to_zyx(mask, mask_nii)
    # Pull voxel sizes (mm) from the affine; nib returns abs of the diagonal.
    aff = raw_nii.affine
    vox_mm = np.abs(np.diag(aff)[:3])  # (x, y, z) in mm
    vox_um = vox_mm * 1000.0
    # Reorder to (z, y, x)
    fr_voxel_um = [float(vox_um[2]), float(vox_um[1]), float(vox_um[0])]

    # crop_fullres = the patch as-is (in z, y, x order)
    crop_fr = raw_zyx.astype(np.uint16)
    mask_fr = mask_zyx.astype(np.uint8)

    # crop_4um = resample crop_fr to 128^3
    zoom_z = PATCH_4UM / crop_fr.shape[0]
    zoom_y = PATCH_4UM / crop_fr.shape[1]
    zoom_x = PATCH_4UM / crop_fr.shape[2]
    crop_4um = ndi_zoom(crop_fr.astype(np.float32), (zoom_z, zoom_y, zoom_x), order=1)
    crop_4um = np.clip(crop_4um, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    # mask_4um = max-pool to 128^3 (preserves sparse positives)
    mask_4um = downsample_mask_to_128(mask_fr)

    # Effective 4um voxel size after resampling:
    eff_vox_4um_um = [
        fr_voxel_um[0] * crop_fr.shape[0] / PATCH_4UM,
        fr_voxel_um[1] * crop_fr.shape[1] / PATCH_4UM,
        fr_voxel_um[2] * crop_fr.shape[2] / PATCH_4UM,
    ]

    return {
        "crop_4um": crop_4um,
        "crop_fullres": crop_fr,
        "mask_4um": mask_4um,
        "mask_fullres": mask_fr,
        "fullres_voxel_um": fr_voxel_um,
        "eff_4um_voxel_um": eff_vox_4um_um,
    }


# ---------------------------------------------------------------------------
# QC: 3-panel mid-slice
# ---------------------------------------------------------------------------

def render_qc_three(crop_4um, crop_fr, mask_fr, out_path: Path, title: str):
    z4 = crop_4um.shape[0] // 2
    zf = crop_fr.shape[0] // 2
    vmax_4 = float(np.percentile(crop_4um, 99.5)) or 1.0
    vmax_f = float(np.percentile(crop_fr, 99.5)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), constrained_layout=True)

    axes[0].imshow(crop_4um[z4], cmap="gray", vmin=0, vmax=vmax_4)
    axes[0].set_title(f"4um axial mid  z={z4}/{crop_4um.shape[0]}  shape={crop_4um.shape}")

    axes[1].imshow(crop_fr[zf], cmap="gray", vmin=0, vmax=vmax_f)
    axes[1].set_title(f"fullres axial mid  z={zf}/{crop_fr.shape[0]}  shape={crop_fr.shape}")

    axes[2].imshow(crop_fr[zf], cmap="gray", vmin=0, vmax=vmax_f)
    mid_mask_slice = mask_fr[zf] if mask_fr.shape[0] > zf else np.zeros_like(crop_fr[zf])
    overlay = np.ma.masked_where(mid_mask_slice == 0, mid_mask_slice)
    axes[2].imshow(overlay, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    pos_mid = int((mid_mask_slice > 0).sum())
    pos_total = int((mask_fr > 0).sum())
    axes[2].set_title(f"fullres + mask  mid+={pos_mid}  total+={pos_total}")

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
    ap.add_argument("--n-total", type=int, default=250)
    ap.add_argument("--ozx-share", type=int, default=150,
                    help="patches drawn from atlas-sampled OZX datasets (rest from v0.5). "
                         "Default 150 follows the per-dataset 2x v0.5 rule: 3 OZX datasets "
                         "x X + v0.5 x 2X = 5X = 250 -> X=50, OZX=150, v0.5=100.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gt-root",
        default=os.environ.get("GT", "/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized"),
    )
    ap.add_argument("--qc-dir", default="qc_images/abeta_v4_pairs")
    args = ap.parse_args()

    n_ozx = args.ozx_share
    n_v05 = args.n_total - n_ozx
    print(f"=== v4 plan ===  total={args.n_total}  OZX={n_ozx}  v0.5={n_v05}\n")

    # OZX-track subjects
    print("--- OZX track subject discovery ---")
    ozx_subjects = []
    for cfg in DATASETS_OZX:
        subs = discover_ozx_subjects(cfg)
        ozx_subjects.extend(subs)
        print(f"  {Path(cfg['root']).name}: {len(subs)} subjects (spimquant={cfg['spimquant']})")
    print(f"  total OZX subjects: {len(ozx_subjects)}")

    ozx_patches = sample_ozx_patches(ozx_subjects, REGION_GROUPS, n_ozx, args.seed)
    print(f"  sampled {len(ozx_patches)} OZX patches")

    # v0.5 track items
    print("\n--- v0.5 track patch enumeration ---")
    v05_all = enumerate_v05_patches(V05_CFG)
    print(f"  found {len(v05_all)} precomputed (raw, mask) pairs across "
          f"{len(set(p['subject_id'] for p in v05_all))} subjects")
    v05_picks = sample_v05_patches(v05_all, n_v05, args.seed + 7)
    print(f"  sampled {len(v05_picks)} v0.5 patches")

    v4_root = Path(args.gt_root) / "Abeta" / "v4"
    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    affine_4um = np.diag([RES_4UM_MM, RES_4UM_MM, RES_4UM_MM, 1.0])

    top_entries = []
    skipped = []

    # ---- OZX track ----
    counts = {r: 0 for r in REGION_GROUPS}
    print("\n--- writing OZX patches ---")
    for entry in ozx_patches:
        roi = entry["region_group"]
        counts[roi] += 1
        patch_id = f"{roi}_{counts[roi]:03d}_{entry['subject_id']}"
        out_dir = v4_root / roi / patch_id
        try:
            res = extract_ozx_record(entry)
        except Exception as exc:
            print(f"  {patch_id}: EXTRACT FAILED ({exc})")
            counts[roi] -= 1
            skipped.append({"patch_id": patch_id, "track": "ozx", "reason": str(exc)})
            continue
        if "skip_reason" in res:
            print(f"  {patch_id}: skip ({res['skip_reason']})")
            counts[roi] -= 1
            skipped.append({"patch_id": patch_id, "track": "ozx", "reason": res["skip_reason"]})
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(res["crop_4um"], affine_4um),
                 str(out_dir / "crop_4um.nii.gz"))
        nib.save(nib.Nifti1Image(res["mask_4um"], affine_4um),
                 str(out_dir / "mask_4um.nii.gz"))
        fr_um = res["fullres_voxel_um"]
        affine_fr = np.diag([fr_um[0] / 1000.0, fr_um[1] / 1000.0, fr_um[2] / 1000.0, 1.0])
        nib.save(nib.Nifti1Image(res["crop_fullres"], affine_fr),
                 str(out_dir / "crop_fullres.nii.gz"))
        nib.save(nib.Nifti1Image(res["mask_fullres"], affine_fr),
                 str(out_dir / "mask_fullres.nii.gz"))

        ds_short = entry["dataset_name"].replace("mouse_app_", "")
        render_qc_three(
            res["crop_4um"], res["crop_fullres"], res["mask_fullres"],
            qc_dir / f"{patch_id}.png",
            title=f"{ds_short}/{entry['subject_id']}  {patch_id}  ROI={roi}  (OZX)",
        )

        top_entries.append({
            "patch_id": patch_id,
            "track": "ozx",
            "roi": roi,
            "subject_id": entry["subject_id"],
            "dataset_name": entry["dataset_name"],
            "dataset_root": entry["dataset_root"],
            "spimquant_subdir": entry["spimquant_subdir"],
            "stain_channel": entry["stain_channel"],
            "center_phys_mm_zyx": entry["center_phys"],
            "center_vox_4um_zyx": entry["center_vox"],
            "center_vox_fullres_zyx": entry["center_vox_fullres"],
            "box_4um_zyx": res["box_4um_zyx"],
            "box_fr_zyx": res["box_fr_zyx"],
            "fullres_voxel_um_zyx": res["fullres_voxel_um"],
            "fullres_zarr": entry["fullres_zarr"],
            "resampled_zarr": entry["zarr_path"],
            "resampled_sidecar": entry["sidecar"],
            "ozx_path": entry["ozx_path"],
            "files": {
                "crop_4um":     str(out_dir / "crop_4um.nii.gz"),
                "crop_fullres": str(out_dir / "crop_fullres.nii.gz"),
                "mask_4um":     str(out_dir / "mask_4um.nii.gz"),
                "mask_fullres": str(out_dir / "mask_fullres.nii.gz"),
            },
        })

    # ---- v0.5 track ----
    v05_root = v4_root / "v0.5"
    print("\n--- writing v0.5 patches ---")
    v05_counts = Counter()
    for item in v05_picks:
        try:
            res = extract_v05_record(item)
        except Exception as exc:
            print(f"  v0.5 {item['subject_id']} {item['label']} p{item['patch_idx']:04d}: "
                  f"EXTRACT FAILED ({exc})")
            skipped.append({"track": "v0.5", "item": item, "reason": str(exc)})
            continue

        v05_counts[item["label"]] += 1
        n_in_label = v05_counts[item["label"]]
        patch_id = (f"v05_{item['label']}_{n_in_label:03d}_"
                    f"{item['subject_id']}_p{item['patch_idx']:04d}")
        out_dir = v05_root / patch_id
        out_dir.mkdir(parents=True, exist_ok=True)

        nib.save(nib.Nifti1Image(res["crop_4um"], affine_4um),
                 str(out_dir / "crop_4um.nii.gz"))
        nib.save(nib.Nifti1Image(res["mask_4um"], affine_4um),
                 str(out_dir / "mask_4um.nii.gz"))
        fr_um = res["fullres_voxel_um"]
        affine_fr = np.diag([fr_um[0] / 1000.0, fr_um[1] / 1000.0, fr_um[2] / 1000.0, 1.0])
        nib.save(nib.Nifti1Image(res["crop_fullres"], affine_fr),
                 str(out_dir / "crop_fullres.nii.gz"))
        nib.save(nib.Nifti1Image(res["mask_fullres"], affine_fr),
                 str(out_dir / "mask_fullres.nii.gz"))

        render_qc_three(
            res["crop_4um"], res["crop_fullres"], res["mask_fullres"],
            qc_dir / f"{patch_id}.png",
            title=f"v0.5/{item['subject_id']}  label={item['label']} p={item['patch_idx']:04d}",
        )

        top_entries.append({
            "patch_id": patch_id,
            "track": "v0.5",
            "roi": "v0.5",
            "label": item["label"],
            "subject_id": item["subject_id"],
            "dataset_name": item["dataset_name"],
            "dataset_root": item["dataset_root"],
            "spimquant_subdir": item["spimquant_subdir"],
            "raw_path_src": item["raw_path"],
            "mask_path_src": item["mask_path"],
            "fullres_voxel_um_zyx": res["fullres_voxel_um"],
            "eff_4um_voxel_um_zyx": res["eff_4um_voxel_um"],
            "files": {
                "crop_4um":     str(out_dir / "crop_4um.nii.gz"),
                "crop_fullres": str(out_dir / "crop_fullres.nii.gz"),
                "mask_4um":     str(out_dir / "mask_4um.nii.gz"),
                "mask_fullres": str(out_dir / "mask_fullres.nii.gz"),
            },
        })

    # ---- Manifest ----
    v4_root.mkdir(parents=True, exist_ok=True)
    ozx_roi_counts = dict(Counter(p["roi"] for p in top_entries if p["track"] == "ozx"))
    v05_label_counts = dict(Counter(p["label"] for p in top_entries if p["track"] == "v0.5"))
    manifest = {
        "config": {
            "stain": "Abeta",
            "n_total_requested": args.n_total,
            "n_ozx_requested": n_ozx,
            "n_v05_requested": n_v05,
            "n_written": len(top_entries),
            "n_skipped": len(skipped),
            "seed": args.seed,
            "patch_size_4um_vox": PATCH_4UM,
            "patch_size_physical_um": PATCH_PHYS_UM,
            "region_groups": {n: {"regions": r, "weight": w}
                              for n, (r, w) in REGION_GROUPS.items()},
            "datasets_ozx": DATASETS_OZX,
            "dataset_v05": V05_CFG,
            "ozx_roi_counts": ozx_roi_counts,
            "v05_label_counts": v05_label_counts,
        },
        "patches": top_entries,
        "skipped": skipped,
    }
    (v4_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n=== summary ===")
    print(f"  wrote {len(top_entries)} patches  (OZX={sum(1 for p in top_entries if p['track']=='ozx')}, "
          f"v0.5={sum(1 for p in top_entries if p['track']=='v0.5')})")
    print(f"  skipped {len(skipped)}")
    print(f"  OZX ROI counts: {ozx_roi_counts}")
    print(f"  v0.5 label counts: {v05_label_counts}")
    print(f"  GT root: {v4_root}")
    print(f"  manifest: {v4_root/'manifest.json'}")
    print(f"  QC: {qc_dir}/")


if __name__ == "__main__":
    main()
