"""finalize_finetune_gold.py — finalise WT-normalized fine-tune masks.

After ``build_finetune_normalized.py`` writes raw + normalized + seg_otsu2 +
seg_otsu3 NIfTI files for every disease patch, you still need to:

  1. Pick a default segmentation variant (otsu2 vs otsu3) — bulk-copy that
     into ``*_seg_gold.nii.gz`` for every patch.

  2. Browse the QC PNGs and override per patch where the other variant is
     clearly better.

  3. Open each patch in napari and hand-delete the obvious false positives
     (mostly boundary artifacts on the brain edge). Save the edited labels
     back over the existing ``*_seg_gold.nii.gz``.

  4. Mark patches as reviewed so you can resume the next day.

  5. Write the final ``manifest_gold.json`` for fine-tune training.

Subcommands:

    init        bulk-copy a chosen otsu variant into *_seg_gold.nii.gz
                Examples:
                  finalize init --use otsu2                  # all patches
                  finalize init --use otsu3 --patch sub-AS40F2_patch01
                  finalize init --use otsu2 --force          # overwrite

    napari      launch napari for one patch with raw + normalized + the two
                otsu candidates + the editable gold mask. Save the gold layer
                from napari (Ctrl+S after selecting it) when done. The path
                is printed so you can paste it into the save dialog.
                  finalize napari --patch sub-AS40F2_patch01

    review      mark a patch as reviewed (sets a flag in
                .review_state.json next to manifest.json):
                  finalize review --patch sub-AS40F2_patch01
                  finalize review --patch sub-AS40F2_patch01 --notes "deleted 2 boundary blobs"
                  finalize review --patch sub-AS40F2_patch01 --unmark

    status      print a table of every patch with: gold-mask exists, voxels in
                the mask, reviewed flag.

    manifest    write ft_normalized/Abeta/manifest_gold.json containing only
                the patches that have been (a) initialized AND (b) reviewed.
                Use --include-unreviewed to include init'd-but-unreviewed too.

Suggested workflow:

    pixi run python scripts/finalize_finetune_gold.py init --use otsu2
    pixi run python scripts/finalize_finetune_gold.py status
    # ... browse QC PNGs in your file manager, override variants where needed
    pixi run python scripts/finalize_finetune_gold.py init --use otsu3 \\
            --patch sub-AS208F2_patch00
    # ... napari pass on each patch
    pixi run python scripts/finalize_finetune_gold.py napari --patch sub-AS40F2_patch01
    pixi run python scripts/finalize_finetune_gold.py review --patch sub-AS40F2_patch01
    # ... when all 46 reviewed
    pixi run python scripts/finalize_finetune_gold.py manifest
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "ft_normalized" / "Abeta"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
GOLD_MANIFEST_PATH = OUTPUT_ROOT / "manifest_gold.json"
REVIEW_STATE_PATH = OUTPUT_ROOT / ".review_state.json"


# ---------------------------------------------------------------------------
# Loading + per-patch path helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> List[Dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(
            f"manifest not found at {MANIFEST_PATH} — run "
            "scripts/build_finetune_normalized.py first."
        )
    with open(MANIFEST_PATH) as f:
        return json.load(f)["patches"]


def _patch_id(entry: Dict) -> str:
    """Stable per-patch ID like sub-AS40F2_patch01 (matches the filename stem)."""
    raw = entry.get("raw_path", "")
    if not raw:
        return f"{entry['subject_id']}_unknown"
    name = Path(raw).name
    if "_crop" in name:
        return name.split("_crop")[0]
    return name.split(".")[0]


def _patch_dir(entry: Dict) -> Path:
    return REPO_ROOT / Path(entry["raw_path"]).parent


def _path_for(entry: Dict, suffix: str) -> Path:
    """e.g. _path_for(entry, 'seg_gold.nii.gz') -> .../sub-XXX_patchNN_seg_gold.nii.gz"""
    return _patch_dir(entry) / f"{_patch_id(entry)}_{suffix}"


def _load_review_state() -> Dict[str, Dict]:
    if REVIEW_STATE_PATH.exists():
        with open(REVIEW_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_review_state(state: Dict) -> None:
    REVIEW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _filter(patches: List[Dict], patch_filter: Optional[str]) -> List[Dict]:
    if patch_filter is None:
        return patches
    out = [p for p in patches if _patch_id(p) == patch_filter]
    if not out:
        sys.exit(
            f"no patch matches '{patch_filter}'. Use 'status' to list valid IDs."
        )
    return out


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    variant = args.use
    suffix_in = f"seg_{variant}.nii.gz"
    suffix_out = "seg_gold.nii.gz"

    patches = _filter(_load_manifest(), args.patch)
    n_done = 0
    n_skip = 0
    n_missing = 0
    state = _load_review_state()

    for p in patches:
        src = _path_for(p, suffix_in)
        dst = _path_for(p, suffix_out)
        pid = _patch_id(p)
        if not src.exists():
            print(f"  MISS  {pid}: {src.name} not found")
            n_missing += 1
            continue
        if dst.exists() and not args.force:
            n_skip += 1
            continue
        shutil.copyfile(src, dst)
        # Track what variant was used to seed the gold mask
        st = state.setdefault(pid, {})
        st["init_variant"] = variant
        st["initialized"] = True
        # Re-init invalidates any prior review
        if args.force:
            st["reviewed"] = False
        n_done += 1
        print(f"  init  {pid} <- {variant}")

    _save_review_state(state)
    print(f"\nDone. initialized={n_done}  skipped(existing)={n_skip}  missing={n_missing}")
    if n_skip:
        print("  (use --force to overwrite already-initialized gold masks)")


# ---------------------------------------------------------------------------
# napari
# ---------------------------------------------------------------------------

def cmd_napari(args: argparse.Namespace) -> None:
    patches = _filter(_load_manifest(), args.patch)
    if len(patches) != 1:
        sys.exit("napari: --patch must select exactly one patch")
    p = patches[0]
    pid = _patch_id(p)
    raw = _path_for(p, "crop128.nii.gz")
    norm = _path_for(p, "normalized128.nii.gz")
    o2 = _path_for(p, "seg_otsu2.nii.gz")
    o3 = _path_for(p, "seg_otsu3.nii.gz")
    gold = _path_for(p, "seg_gold.nii.gz")

    if not gold.exists():
        sys.exit(
            f"gold mask not initialized for {pid}. Run:\n"
            f"  finalize init --use otsu2 --patch {pid}"
        )

    print(f"Opening napari for {pid}")
    print(f"  dataset:    {p['dataset']}")
    print(f"  region:     {p.get('region_group', '?')}")
    print(f"  mu_wt:      {p['mu_wt']:.1f}    sigma_wt: {p['sigma_wt']:.1f}")
    print(f"  k2 components: {p['n_components_otsu2']}    k3: {p['n_components_otsu3']}")
    if p.get("wt_fallback_count", 0):
        print(f"  USED WT GLOBAL FALLBACK ({p['wt_fallback_count']}/{len(p.get('wt_references', []))})")
    print(f"  gold mask file (save edits here):")
    print(f"    {gold}")
    print()

    # Launch napari in this process so the user can edit interactively.
    import napari

    raw_img = nib.load(str(raw)).get_fdata()
    norm_img = nib.load(str(norm)).get_fdata()
    o2_img = nib.load(str(o2)).get_fdata().astype(np.uint8)
    o3_img = nib.load(str(o3)).get_fdata().astype(np.uint8)
    gold_img = nib.load(str(gold)).get_fdata().astype(np.uint8)

    viewer = napari.Viewer(title=f"{pid}  ({p['dataset']})")
    viewer.add_image(raw_img, name="raw", colormap="gray")
    viewer.add_image(
        norm_img, name="normalized (z-score)", colormap="magma",
        contrast_limits=(float(np.percentile(norm_img, 2)),
                         float(np.percentile(norm_img, 99.7))),
        visible=False,
    )
    viewer.add_labels(o2_img, name="otsu2 (ref)", visible=False, opacity=0.5)
    viewer.add_labels(o3_img, name="otsu3 (ref)", visible=False, opacity=0.5)
    gold_layer = viewer.add_labels(gold_img, name="GOLD (edit me)", opacity=0.6)
    viewer.layers.selection.active = gold_layer

    # Bind a save shortcut to write straight back to disk so the user doesn't
    # have to fiddle with the file dialog.
    @viewer.bind_key("Ctrl-S", overwrite=True)
    def save_gold(_v):
        out = nib.Nifti1Image(
            gold_layer.data.astype(np.uint8),
            nib.load(str(gold)).affine,
        )
        nib.save(out, str(gold))
        viewer.status = f"saved gold mask -> {gold.name}"
        print(f"saved {gold}")

    print("napari hotkeys:")
    print("  Ctrl+S       save the GOLD layer back to disk")
    print("  e/p          eraser / paint brush on the labels layer")
    print("  q            close napari window")
    print()
    print("After saving, mark the patch reviewed:")
    print(f"  pixi run python scripts/finalize_finetune_gold.py review --patch {pid}")
    napari.run()


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> None:
    patches = _filter(_load_manifest(), args.patch)
    state = _load_review_state()
    for p in patches:
        pid = _patch_id(p)
        st = state.setdefault(pid, {})
        if args.unmark:
            st["reviewed"] = False
            st.pop("review_notes", None)
            print(f"  unmark  {pid}")
        else:
            st["reviewed"] = True
            if args.notes:
                st["review_notes"] = args.notes
            print(f"  review  {pid}" + (f" ({args.notes})" if args.notes else ""))
    _save_review_state(state)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    patches = _load_manifest()
    state = _load_review_state()

    n_total = len(patches)
    n_init = 0
    n_reviewed = 0

    print(f"{'patch_id':<28} {'dataset':<32} {'region':<13} {'gold':>10} "
          f"{'k2':>5} {'k3':>5} init  reviewed  notes")
    print("-" * 130)
    for p in sorted(patches, key=lambda q: (q["dataset"], _patch_id(q))):
        pid = _patch_id(p)
        st = state.get(pid, {})
        gold = _path_for(p, "seg_gold.nii.gz")
        if gold.exists():
            try:
                n_vox = int(nib.load(str(gold)).get_fdata().astype(bool).sum())
            except Exception:
                n_vox = -1
            gold_str = f"{n_vox:>10d}"
            n_init += 1
        else:
            gold_str = f"{'(none)':>10}"
        if st.get("reviewed"):
            n_reviewed += 1
        init_var = st.get("init_variant", "-")
        rev = "yes" if st.get("reviewed") else "no"
        notes = st.get("review_notes", "")
        fb_tag = " *" if p.get("wt_fallback_count", 0) > 0 else "  "
        print(
            f"{pid:<28} {p['dataset']:<32} {p.get('region_group','?'):<13} "
            f"{gold_str} {p['n_components_otsu2']:>5} {p['n_components_otsu3']:>5}  "
            f"{init_var:<5} {rev:<8} {notes}{fb_tag}"
        )

    print("-" * 130)
    print(
        f"  total {n_total}  initialized {n_init}  reviewed {n_reviewed}  "
        f"remaining {n_total - n_reviewed}"
    )
    print("  '*' = patch used WT global fallback (no patch-matched stats)")


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    patches = _load_manifest()
    state = _load_review_state()

    out_patches = []
    for p in patches:
        pid = _patch_id(p)
        st = state.get(pid, {})
        gold = _path_for(p, "seg_gold.nii.gz")
        if not gold.exists():
            continue
        if not st.get("reviewed") and not args.include_unreviewed:
            continue
        entry = dict(p)  # copy
        entry["seg_gold_path"] = str(gold.relative_to(REPO_ROOT))
        entry["init_variant"] = st.get("init_variant")
        entry["reviewed"] = bool(st.get("reviewed"))
        if st.get("review_notes"):
            entry["review_notes"] = st["review_notes"]
        out_patches.append(entry)

    src = json.load(open(MANIFEST_PATH))
    out = {
        "config": {
            **src.get("config", {}),
            "include_unreviewed": bool(args.include_unreviewed),
            "n_gold_patches": len(out_patches),
        },
        "wt_excluded": src.get("wt_excluded", []),
        "patches": out_patches,
    }
    with open(GOLD_MANIFEST_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {GOLD_MANIFEST_PATH} with {len(out_patches)} patches")
    if not args.include_unreviewed:
        n_skipped = sum(
            1 for p in patches
            if _path_for(p, "seg_gold.nii.gz").exists()
            and not state.get(_patch_id(p), {}).get("reviewed")
        )
        if n_skipped:
            print(
                f"  (skipped {n_skipped} initialized but unreviewed patches; "
                f"use --include-unreviewed to bundle them too)"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="bulk-copy an otsu variant to seg_gold.nii.gz")
    p_init.add_argument("--use", choices=["otsu2", "otsu3"], required=True)
    p_init.add_argument("--patch", default=None, help="restrict to a single patch_id")
    p_init.add_argument("--force", action="store_true", help="overwrite existing gold masks")
    p_init.set_defaults(func=cmd_init)

    p_nap = sub.add_parser("napari", help="open one patch in napari for hand editing")
    p_nap.add_argument("--patch", required=True)
    p_nap.set_defaults(func=cmd_napari)

    p_rev = sub.add_parser("review", help="mark a patch as reviewed")
    p_rev.add_argument("--patch", required=True)
    p_rev.add_argument("--notes", default=None)
    p_rev.add_argument("--unmark", action="store_true")
    p_rev.set_defaults(func=cmd_review)

    p_st = sub.add_parser("status", help="print a per-patch status table")
    p_st.set_defaults(func=cmd_status)

    p_mf = sub.add_parser("manifest", help="write manifest_gold.json")
    p_mf.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="also include patches that have a gold mask but no review flag",
    )
    p_mf.set_defaults(func=cmd_manifest)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
