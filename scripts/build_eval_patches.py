"""Build the simclr_eval evaluation set.

Two things, organized under ../simclr_eval/ (i.e. alongside the Lumivox repo):

  (1) val_patches/  — the fixed-seed validation patches (the same 33 the sweep
      held out, seed-42 subject split) with their human-reviewed GT, copied as:
          ../simclr_eval/val_patches/<patch_id>/crop_4um.nii.gz
          ../simclr_eval/val_patches/<patch_id>/mask_4um.nii.gz   (GT)

  (2) sub-<ID>/      — new 1024^3 eval crops from the 4um images, 2 cortex + 1
      hippocampus per subject, each with its SPIMquant OZX mask downsampled to
      the same 1024^3 grid:
          ../simclr_eval/sub-<ID>/00N.nii.gz            (crop, run inference on this)
          ../simclr_eval/sub-<ID>/00N_ozxmask.nii.gz    (OZX reference mask)

Usage:
    pixi run python scripts/build_eval_patches.py --val-only      # (1) only — light, login-node OK
    pixi run python scripts/build_eval_patches.py                 # (1) + (2) — heavy, run on a compute node

The 1024^3 extraction reads a ~4mm^3 full-res region per OZX mask; it is
downsampled to 1024^3 by streaming z-slabs (block-max, sparse-preserving) so peak
memory stays ~1-2 GB rather than materialising tens of GB.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

EVAL_ROOT = Path("../simclr_eval")
MANIFEST = "manifests/abeta_ft_v4_A.json"
PATCH = 1024
RES_4UM_MM = 0.004
PATCH_PHYS_UM = PATCH * RES_4UM_MM * 1000.0  # 4096 um

# 2 cortex + 1 hippocampus per subject.
ROI_PLAN = [("cortex", ["L_Isocortex", "R_Isocortex"]),
            ("cortex", ["L_Isocortex", "R_Isocortex"]),
            ("hippocampus", ["L_Hippocampal formation", "R_Hippocampal formation"])]

SUBJECTS = {
    "sub-AS40F2":  dict(root="/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2",
                        spimquant="spimquant-v0.6.0rc2_84a605e_ozx"),
    "sub-AS161F3": dict(root="/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3",
                        spimquant="spimquant-v0.6.0rc2_84a605e_ozx"),
    "sub-AS176F3": dict(root="/nfs/trident3/lightsheet/prado/mouse_app_vaccine_batch1",
                        spimquant="spimquant_v0.6.0rc1"),
}


# --------------------------------------------------------------------------- #
# (1) Validation patches
# --------------------------------------------------------------------------- #

def organize_val_patches(seed: int = 42, val_fraction: float = 0.2):
    data = json.loads(Path(MANIFEST).read_text())
    entries = data["patches"]
    subjects = sorted({e["subject_id"] for e in entries})
    n_val = max(1, round(len(subjects) * val_fraction))
    val = set(random.Random(seed).sample(subjects, n_val))
    vpatches = [e for e in entries if e["subject_id"] in val]
    out_root = EVAL_ROOT / "val_patches"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[val] {len(vpatches)} patches from {sorted(val)}")
    n = 0
    for e in vpatches:
        d = out_root / e["patch_id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e["raw_path"], d / "crop_4um.nii.gz")
        shutil.copy2(e["seg_gold_path"], d / "mask_4um.nii.gz")
        n += 1
    print(f"[val] wrote {n} -> {out_root}")


# --------------------------------------------------------------------------- #
# (2) 1024^3 crops + OZX masks
# --------------------------------------------------------------------------- #

def _discover(sub: str, cfg: dict):
    """Resolve 4um zarr, fullres zarr, OZX mask, dseg + labels for a subject."""
    import re
    root = Path(cfg["root"])
    sq = root / "derivatives" / cfg["spimquant"]
    rs = root / "bids" / "derivatives" / "resampled" / sub / "micr"
    fr = root / "bids" / sub / "micr"

    def prefer(paths):
        std = [p for p in paths if "45deg" not in p.name and "90deg" not in p.name]
        return (std or paths)

    zarr4 = prefer(sorted(rs.glob("*_res-4um_SPIM.ome.zarr")))
    frz = prefer(sorted(fr.glob("*_SPIM.ome.zarr")))
    ozx = sorted((sq / sub / "micr").glob("*stain-Abeta_level-0*mask.ozx"))
    dseg = sorted((sq / sub / "micr").glob("*seg-roi22_from-ABAv3_*dseg.nii.gz"))
    if not dseg:  # broaden the search
        dseg = sorted((sq / sub).rglob("*seg-roi22*dseg.nii.gz"))
    tsv = sorted((sq / sub / "micr").glob("*seg-roi22_from-ABAv3_*dseg.tsv"))
    ref_tsv = Path(__file__).resolve().parent.parent / "lumivox/data/reference/seg-roi22_dseg.tsv"
    labels = str(tsv[0]) if tsv else (str(ref_tsv) if ref_tsv.exists() else None)

    missing = [n for n, v in [("4um", zarr4), ("fullres", frz), ("ozx", ozx),
                              ("dseg", dseg)] if not v]
    return dict(
        zarr4=str(zarr4[0]) if zarr4 else None,
        fullres=str(frz[0]) if frz else None,
        sidecar=str(zarr4[0]) + ".json" if zarr4 else None,
        ozx=str(ozx[0]) if ozx else None,
        dseg=str(dseg[0]) if dseg else None,
        labels=labels,
        missing=missing,
    )


def _extract_ozx_1024(mask_darr, cv_fr, pz, py, px, out=PATCH, zchunk=32):
    """Block-max downsample the full-res OZX box to out^3, streaming z-slabs."""
    Z, Y, X = mask_darr.shape
    fz0 = int(round(cv_fr[0] - pz / 2.0))
    fy0 = int(round(cv_fr[1] - py / 2.0))
    fx0 = int(round(cv_fr[2] - px / 2.0))
    facz, facy, facx = pz / out, py / out, px / out
    res = np.zeros((out, out, out), dtype=np.uint8)
    siy0, siy1 = max(0, fy0), min(Y, fy0 + py)
    six0, six1 = max(0, fx0), min(X, fx0 + px)
    if siy1 <= siy0 or six1 <= six0:
        return res
    for oz in range(0, out, zchunk):
        oz1 = min(out, oz + zchunk)
        iz0 = max(0, fz0 + int(round(oz * facz)))
        iz1 = min(Z, fz0 + int(round(oz1 * facz)))
        if iz1 <= iz0:
            continue
        slab = np.asarray(mask_darr[iz0:iz1, siy0:siy1, six0:six1].compute()) > 0
        nz = np.argwhere(slab)
        if nz.size == 0:
            continue
        bz = (iz0 - fz0) + nz[:, 0]
        by = (siy0 - fy0) + nz[:, 1]
        bx = (six0 - fx0) + nz[:, 2]
        rz = np.minimum((bz / facz).astype(np.int64), out - 1)
        ry = np.minimum((by / facy).astype(np.int64), out - 1)
        rx = np.minimum((bx / facx).astype(np.int64), out - 1)
        res[rz, ry, rx] = 1
    return res


def build_1024_patches(seed: int = 1234):
    import zarr
    from zarrnii import ZarrNii, ZarrNiiAtlas
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lumivox.data.manifest import resolve_stain_channel

    def open_zarr(p):
        p = Path(p)
        if (p / "0" / ".zarray").exists():
            return zarr.open(str(p / "0"), mode="r")
        if (p / "s0").exists():
            return zarr.open(str(p / "s0"), mode="r")
        return zarr.open(str(p), mode="r")[0]

    def fr_vox_um(frz):
        meta = json.loads((Path(frz) / ".zattrs").read_text())
        sc = meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        return [sc[1] * 1000.0, sc[2] * 1000.0, sc[3] * 1000.0]

    affine_4um = np.diag([RES_4UM_MM] * 3 + [1.0])
    rng = np.random.default_rng(seed)

    for sub, cfg in SUBJECTS.items():
        info = _discover(sub, cfg)
        print(f"\n=== {sub} ===")
        if info["missing"]:
            print(f"  SKIP — missing {info['missing']} (e.g. dseg unavailable). "
                  f"Cannot atlas-sample ROIs for this subject.")
            continue
        atlas = ZarrNiiAtlas.from_files(dseg_path=info["dseg"], labels_path=info["labels"])
        sc = json.loads(Path(info["sidecar"]).read_text())["fullres_to_resampled"]["scale"]
        res_scale = np.array([sc["z"], sc["y"], sc["x"]])
        fr_inv = ZarrNii.from_ome_zarr(info["fullres"], channels=[0]).affine.invert()
        stain_ch = resolve_stain_channel(info["zarr4"], "Abeta", info["fullres"])
        r4 = open_zarr(info["zarr4"])
        _, Z4, Y4, X4 = r4.shape
        fr_um = fr_vox_um(info["fullres"])
        pz = int(round(PATCH_PHYS_UM / fr_um[0]))
        py = int(round(PATCH_PHYS_UM / fr_um[1]))
        px = int(round(PATCH_PHYS_UM / fr_um[2]))
        znmask = ZarrNii.from_ome_zarr(info["ozx"], channels=[0])
        mask_darr = znmask.darr[0]

        out_dir = EVAL_ROOT / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = 0
        for roi_name, region_ids in ROI_PLAN:
            centers = atlas.sample_region_patches(
                n_patches=1, region_ids=region_ids, seed=int(rng.integers(0, 2**31)))
            if not len(centers):
                print(f"  {roi_name}: no center sampled, skipping")
                continue
            c = centers[0]
            vox_fr = fr_inv @ np.array(c)
            cv = vox_fr * res_scale
            # 1024^3 = 4mm is large vs the brain (Z ~5mm); cortex/hippo centers sit
            # near the surface and would clip. Clamp/shift the box to stay in-bounds
            # (ROI still inside the patch, just off-centre), then re-derive the OZX
            # centre from the shifted box so crop and mask stay aligned.
            if Z4 < PATCH or Y4 < PATCH or X4 < PATCH:
                print(f"  {roi_name}: volume {(Z4,Y4,X4)} smaller than {PATCH}^3, skipping")
                continue
            def clamp(cc, dim):
                lo = max(0, min(int(round(cc - PATCH / 2)), dim - PATCH))
                return lo, lo + PATCH
            z0, z1 = clamp(cv[0], Z4)
            y0, y1 = clamp(cv[1], Y4)
            x0, x1 = clamp(cv[2], X4)
            cv_new = np.array([(z0 + z1) / 2.0, (y0 + y1) / 2.0, (x0 + x1) / 2.0])
            cv_fr_new = cv_new / res_scale  # shifted centre -> fullres voxel for OZX
            idx += 1
            tag = f"{idx:03d}"
            crop = np.asarray(r4[stain_ch, z0:z1, y0:y1, x0:x1], dtype=np.uint16)
            ozx = _extract_ozx_1024(mask_darr, cv_fr_new, pz, py, px)
            nib.save(nib.Nifti1Image(crop, affine_4um), str(out_dir / f"{tag}.nii.gz"))
            nib.save(nib.Nifti1Image(ozx, affine_4um), str(out_dir / f"{tag}_ozxmask.nii.gz"))
            print(f"  {tag} ({roi_name}): crop {crop.shape}  ozx+ voxels={int(ozx.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true",
                    help="organize the validation patches only (light; skip 1024^3 extraction)")
    args = ap.parse_args()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    organize_val_patches()
    if not args.val_only:
        build_1024_patches()
    print("\nDone.")


if __name__ == "__main__":
    main()
