"""Build the v4 fine-tuning manifest from accepted (A) review markers.

Joins the v4 GT manifest (…/Abeta/v4/manifest.json) with the per-patch review
markers Arshya dropped into each patch dir during manual QC:

    A / A.txt     -> ACCEPT  (mask is good, edits already baked into mask_4um)
    R / R.txt     -> REJECT
    M / M*.txt     -> MAYBE   (uncertain; NOT promoted to accept here)
    empty          -> mask intentionally all-zero (true negative)

Only *clean* accepts go into the training manifest: a patch dir that has an A
marker and NO competing R/M marker, and whose name is not an ambiguous edge
case (A_empty, empty_A, drop_edges_too, M_wip, …). Anything ambiguous is
reported, not silently included — so "every A in the manifest is really an A".

Override: v0.5 patches #093-100 are forced to REJECT (not yet reviewed).

Output entry schema matches lumivox/data/dataset_finetune.py:
    raw_path, seg_gold_path, subject_id, dataset, patch_id  (+ track/roi meta)

Usage:
    pixi run python scripts/build_v4_ft_manifest.py \
        --v4-root /nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized/Abeta/v4 \
        --out manifests/abeta_ft_v4_A.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

ACCEPT_NAMES = {"A", "A.txt"}
REJECT_NAMES = {"R", "R.txt"}
MAYBE_PREFIXES = ("M",)          # M, M.txt, M_wip.txt
EMPTY_NAMES = {"empty"}
# Ambiguous / compound markers we refuse to auto-classify.
EDGE_NAMES = {"A_empty", "empty_A", "drop_edges_too"}
SKIP_FILES = {"rebuild_mask_4um.py", "seg_nrrd_to_binary_nifti.py", "view_fullres.sh"}

V05_IDX_RE = re.compile(r"^(\d{3})_v05_")


def resolve_patch_dir(entry: Dict, v4_root: Path) -> Tuple[Path, Path, Path]:
    """Return (patch_dir, crop_4um, mask_4um), resolving stale paths.

    The v0.5 patch dirs were renamed with a leading NNN_ index after the v4
    manifest was written, so manifest['files'] paths are stale for that track.
    Fall back to a basename suffix match within the track's parent dir.
    """
    crop = Path(entry["files"]["crop_4um"])
    pdir = crop.parent
    if pdir.exists():
        return pdir, crop, Path(entry["files"]["mask_4um"])

    base = pdir.name  # original (pre-rename) dir name == patch_id
    parent = pdir.parent
    if not parent.exists():
        parent = v4_root / ("v0.5" if entry.get("track") == "v0.5"
                            else (entry.get("roi") or ""))
    if parent.exists():
        cands = [c for c in parent.iterdir()
                 if c.is_dir() and c.name.endswith(base)]
        if cands:
            real = sorted(cands)[0]
            return real, real / "crop_4um.nii.gz", real / "mask_4um.nii.gz"
    return pdir, crop, Path(entry["files"]["mask_4um"])  # genuinely missing


def _markers_in(patch_dir: Path) -> List[str]:
    """Return marker-ish filenames in a patch dir (excludes data + helper scripts)."""
    out = []
    for p in patch_dir.iterdir():
        if not p.is_file():
            continue
        n = p.name
        if n in SKIP_FILES:
            continue
        if n.endswith(".nii.gz") or n.endswith(".nrrd"):
            continue
        out.append(n)
    return out


def classify(patch_dir: Path, force_reject: bool) -> Tuple[str, List[str], str]:
    """Return (status, markers, note).

    status in {accept, reject, maybe, empty, unreviewed, conflict, edge}.
    """
    markers = _markers_in(patch_dir)
    if force_reject:
        return "reject", markers, "v0.5 #093-100 (forced reject: unreviewed)"

    has_accept = any(m in ACCEPT_NAMES for m in markers)
    has_reject = any(m in REJECT_NAMES for m in markers)
    has_maybe = any(m.split(".")[0].startswith(MAYBE_PREFIXES) and m not in ACCEPT_NAMES
                    for m in markers)
    has_empty = any(m in EMPTY_NAMES for m in markers)
    has_edge = any(m in EDGE_NAMES for m in markers)

    if not markers:
        return "unreviewed", markers, ""
    if has_edge:
        return "edge", markers, "ambiguous compound marker"
    if has_accept and (has_reject or has_maybe):
        return "conflict", markers, "A coexists with R/M"
    if has_accept:
        return "accept", markers, _read_note(patch_dir)
    if has_reject:
        return "reject", markers, _read_note(patch_dir)
    if has_maybe:
        return "maybe", markers, _read_note(patch_dir)
    if has_empty:
        return "empty", markers, ""
    return "unreviewed", markers, ""


def _read_note(patch_dir: Path) -> str:
    for cand in ("A.txt", "R.txt", "M.txt", "M_wip.txt"):
        p = patch_dir / cand
        if p.exists() and p.stat().st_size > 0:
            return p.read_text(errors="replace").strip().replace("\n", " | ")
    return ""


def mask_pos(mask_path: Path) -> Optional[int]:
    try:
        arr = np.asarray(nib.load(str(mask_path)).dataobj)
        return int((arr > 0).sum())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v4-root",
        default="/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized/Abeta/v4",
    )
    ap.add_argument("--out", default="manifests/abeta_ft_v4_A.json")
    ap.add_argument("--count-pos", action="store_true",
                    help="Load each accepted mask to report positive-voxel stats (slower).")
    args = ap.parse_args()

    v4_root = Path(args.v4_root)
    src_manifest = json.loads((v4_root / "manifest.json").read_text())
    src_patches = src_manifest["patches"]

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    accepted: List[Dict] = []
    empty_accepts = 0

    for entry in src_patches:
        patch_dir, crop, mask = resolve_patch_dir(entry, v4_root)
        patch_id = entry["patch_id"]

        force_reject = False
        if entry.get("track") == "v0.5":
            m = V05_IDX_RE.match(patch_dir.name)
            if m and 93 <= int(m.group(1)) <= 100:
                force_reject = True

        if not patch_dir.exists():
            buckets["missing"].append({"patch_id": patch_id, "dir": str(patch_dir)})
            continue

        status, markers, note = classify(patch_dir, force_reject)
        rec = {"patch_id": patch_id, "subject_id": entry["subject_id"],
               "track": entry.get("track"), "roi": entry.get("roi"),
               "markers": markers, "note": note, "dir": str(patch_dir)}
        buckets[status].append(rec)

        if status == "accept":
            npos = mask_pos(mask) if args.count_pos else None
            if npos == 0:
                empty_accepts += 1
            accepted.append({
                "patch_id": patch_id,
                "subject_id": entry["subject_id"],
                "dataset": entry.get("dataset_name", entry.get("track", "v4")),
                "track": entry.get("track"),
                "roi": entry.get("roi"),
                "label": entry.get("label"),
                "raw_path": str(crop),
                "seg_gold_path": str(mask),
                "note": note,
                **({"mask_pos_vox": npos} if npos is not None else {}),
            })

    # ---- manifest ----
    out_manifest = {
        "config": {
            "source": "v4 + manual A markers",
            "v4_root": str(v4_root),
            "stain": "Abeta",
            "crop_size": 128,
            "resolution": "4um (crop_4um/mask_4um)",
            "selection": "clean A markers only; M/R/edge/unreviewed excluded; "
                         "v0.5 #093-100 forced reject",
            "n_accepted": len(accepted),
            "n_empty_accepts": empty_accepts if args.count_pos else None,
        },
        "patches": accepted,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2))

    # ---- report ----
    def n(k):
        return len(buckets.get(k, []))

    print("=" * 64)
    print(f"v4 review tally ({len(src_patches)} patches)")
    print("=" * 64)
    for k in ("accept", "reject", "maybe", "empty", "unreviewed", "conflict",
              "edge", "missing"):
        print(f"  {k:11s}: {n(k)}")
    print(f"\n  -> wrote {len(accepted)} ACCEPTED patches to {out_path}")

    subj = Counter(a["subject_id"] for a in accepted)
    trk = Counter(a["track"] for a in accepted)
    print(f"\n  accepted by track:   {dict(trk)}")
    print(f"  accepted subjects:   {len(subj)}  "
          f"(min/median/max patches per subj: "
          f"{min(subj.values())}/{int(np.median(list(subj.values())))}/{max(subj.values())})")
    if args.count_pos:
        print(f"  accepted empty masks (negatives): {empty_accepts}")

    # Things that need the user's eyes:
    for k, title in (("conflict", "CONFLICTS (A + R/M in same dir) — excluded"),
                     ("edge", "EDGE-NAMED markers — excluded"),
                     ("maybe", "MAYBE (M) — excluded")):
        rows = buckets.get(k, [])
        if rows:
            print(f"\n  --- {title} ({len(rows)}) ---")
            for r in rows:
                print(f"    {r['patch_id']:42s} {r['markers']}  {r['note'][:60]}")

    unrev = [r for r in buckets.get("unreviewed", [])]
    if unrev:
        print(f"\n  --- UNREVIEWED ({len(unrev)}) — no marker present ---")
        for r in unrev:
            print(f"    {r['patch_id']:42s} ({r['track']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
