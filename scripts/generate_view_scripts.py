"""Generate per-patch napari view scripts for the WT-normalized fine-tune data.

For every patch in ``ft_normalized/Abeta/manifest.json`` this writes a tiny
bash script next to the patch files:

  ft_normalized/Abeta/{dataset}/{subject}/view_{patch_id}.sh

Each script `cd`s to its own location and launches napari with:

  - raw 128^3 (gray, default visible)
  - normalized z-score (magma, hidden by default — toggle to inspect contrast)
  - otsu k=2 labels  (visible)
  - otsu k=3 labels  (hidden — toggle to compare)
  - GOLD labels      (only if seg_gold.nii.gz exists; hidden by default)

Usage:
  pixi run python scripts/generate_view_scripts.py          # write all 46
  pixi run python scripts/generate_view_scripts.py --force  # overwrite existing
  pixi run python scripts/generate_view_scripts.py --list   # don't write, just print

Then invoke from anywhere:
  bash ft_normalized/Abeta/.../sub-AS40F2/view_sub-AS40F2_patch00.sh
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "ft_normalized" / "Abeta" / "manifest.json"


SCRIPT_TEMPLATE = """#!/usr/bin/env bash
# Auto-generated view script for {patch_id}
#   dataset:  {dataset}
#   region:   {region}
#   mu_wt:    {mu:.2f}    sigma_wt: {sigma:.2f}
#   k2 components: {k2}    k3 components: {k3}{fallback_note}
#
# Layers loaded into napari:
#   raw          : {raw_name}
#   normalized   : {norm_name}                (hidden by default)
#   otsu2        : {o2_name}            (visible)
#   otsu3        : {o3_name}            (hidden - toggle to compare)
#   GOLD         : {gold_name}             (only if it exists)
#
# Toggle layer visibility in the napari left panel. Use `2`/`3` keyboard
# shortcuts after clicking the layer to enable paint/eraser tools.

set -e
cd "$(dirname "$0")"

python <<'PY'
import os
import nibabel as nib
import numpy as np
import napari

PATCH_ID = "{patch_id}"
RAW = "{raw_name}"
NORM = "{norm_name}"
O2 = "{o2_name}"
O3 = "{o3_name}"
GOLD = "{gold_name}"

raw_img = nib.load(RAW).get_fdata()
norm_img = nib.load(NORM).get_fdata()
o2_img = nib.load(O2).get_fdata().astype(int)
o3_img = nib.load(O3).get_fdata().astype(int)

v = napari.Viewer(title=f"{{PATCH_ID}}  ({dataset_short}, {region_short})")
v.add_image(raw_img, name="raw", colormap="gray")
v.add_image(
    norm_img, name="normalized (z-score)", colormap="magma",
    contrast_limits=(float(np.percentile(norm_img, 2)),
                     max(4.0, float(np.percentile(norm_img, 99.7)))),
    visible=False,
)
v.add_labels(o2_img, name="otsu2", opacity=0.5)
v.add_labels(o3_img, name="otsu3", opacity=0.5, visible=False)

if os.path.exists(GOLD):
    gold_img = nib.load(GOLD).get_fdata().astype(int)
    gold_layer = v.add_labels(gold_img, name="GOLD (edit me)", opacity=0.6)
    v.layers.selection.active = gold_layer

    @v.bind_key("Ctrl-S", overwrite=True)
    def save_gold(_v):
        out = nib.Nifti1Image(
            gold_layer.data.astype(np.uint8),
            nib.load(GOLD).affine,
        )
        nib.save(out, GOLD)
        v.status = f"saved gold mask -> {{GOLD}}"
        print(f"saved {{GOLD}}")
    print("Ctrl+S in napari to save the GOLD layer back to disk.")

napari.run()
PY
"""


def patch_id_from_entry(entry: dict) -> str:
    raw = entry.get("raw_path", "")
    if not raw:
        return f"{entry.get('subject_id','unknown')}_unknown"
    name = Path(raw).name
    if "_crop" in name:
        return name.split("_crop")[0]
    return name.split(".")[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="overwrite existing view scripts")
    p.add_argument("--list", action="store_true", help="just print what would be written")
    args = p.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    n_written = 0
    n_skipped = 0
    written_paths = []

    for entry in manifest["patches"]:
        patch_id = patch_id_from_entry(entry)
        patch_dir = REPO_ROOT / Path(entry["raw_path"]).parent
        if not patch_dir.exists():
            print(f"  WARN: patch dir missing: {patch_dir}")
            continue

        raw_name = f"{patch_id}_crop128.nii.gz"
        norm_name = f"{patch_id}_normalized128.nii.gz"
        o2_name = f"{patch_id}_seg_otsu2.nii.gz"
        o3_name = f"{patch_id}_seg_otsu3.nii.gz"
        gold_name = f"{patch_id}_seg_gold.nii.gz"

        script_path = patch_dir / f"view_{patch_id}.sh"

        fb = entry.get("wt_fallback_count", 0)
        fallback_note = (
            f"\n#   NOTE: this patch used WT global fallback ({fb} ref(s))"
            if fb else ""
        )

        body = SCRIPT_TEMPLATE.format(
            patch_id=patch_id,
            dataset=entry.get("dataset", "?"),
            dataset_short=entry.get("dataset", "?").replace("mouse_app_", ""),
            region=entry.get("region_group", "?"),
            region_short=entry.get("region_group", "?"),
            mu=float(entry.get("mu_wt", 0.0)),
            sigma=float(entry.get("sigma_wt", 0.0)),
            k2=int(entry.get("n_components_otsu2", 0)),
            k3=int(entry.get("n_components_otsu3", 0)),
            fallback_note=fallback_note,
            raw_name=raw_name,
            norm_name=norm_name,
            o2_name=o2_name,
            o3_name=o3_name,
            gold_name=gold_name,
        )

        if args.list:
            print(f"  would write {script_path}")
            n_written += 1
            continue

        if script_path.exists() and not args.force:
            n_skipped += 1
            continue

        script_path.write_text(body)
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written_paths.append(script_path)
        n_written += 1

    if not args.list:
        # Top-level index that lists every per-patch view script with key info,
        # so the user can scan it and copy a path to invoke.
        index_path = REPO_ROOT / "ft_normalized" / "Abeta" / "view_index.txt"
        with open(index_path, "w") as f:
            f.write("# Per-patch napari view scripts\n")
            f.write("# Generated by scripts/generate_view_scripts.py\n")
            f.write("# Invoke any line by running it with bash, e.g.\n")
            f.write("#   bash ft_normalized/Abeta/.../view_sub-AS40F2_patch00.sh\n\n")
            for entry in manifest["patches"]:
                pid = patch_id_from_entry(entry)
                pdir = REPO_ROOT / Path(entry["raw_path"]).parent
                rel = (pdir / f"view_{pid}.sh").relative_to(REPO_ROOT)
                fb_tag = " [WT-fallback]" if entry.get("wt_fallback_count", 0) else ""
                f.write(
                    f"bash {rel}    "
                    f"# {entry.get('dataset','?')} / {entry.get('region_group','?')} "
                    f"k2={entry.get('n_components_otsu2',0)} k3={entry.get('n_components_otsu3',0)}{fb_tag}\n"
                )
        print(f"\nwrote per-patch view scripts: {n_written}, skipped: {n_skipped}")
        print(f"index: {index_path}")
        print("\nrun any patch with:  bash ft_normalized/Abeta/.../view_<patch_id>.sh")
    else:
        print(f"\n(would write {n_written} scripts)")


if __name__ == "__main__":
    main()
