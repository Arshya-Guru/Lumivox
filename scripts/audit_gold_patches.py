"""audit_gold_patches.py — read-only diagnostic sweep over gold-labeled patches.

Scans every ``*_seg_gold.nii.gz`` under ``ft_normalized/Abeta/`` (via manifest
AND tree walk, so nothing slips through) and computes distribution stats,
per-component brightness, and size bins for each. Applies a short set of
flagging rules aimed at catching over-segmented patches before they're used
as fine-tune training targets.

Rationale for the flags:

  Heavy-tailed normalized distributions (norm p99 low but max very high)
  indicate that only a few voxels are truly "plaque-bright" relative to the
  WT reference; a threshold landing near p99 would label ~1% of the volume
  as plaque, but if the gold mask labels >2% the threshold almost certainly
  landed in the bulk (noise) rather than the tail. Similarly, components
  with mean normalized values <1 aren't confidently brighter than WT and
  shouldn't be gold plaque; and components whose mean raw intensity is not
  meaningfully above their local 15-voxel neighborhood aren't visually
  convincing plaques either.

Output: ``ft_normalized/Abeta/gold_audit.txt``. Flagged patches are listed
first (most flags = highest priority), followed by clean ones.

Usage:
    pixi run python scripts/audit_gold_patches.py
    pixi run python scripts/audit_gold_patches.py --out custom_path.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cc3d
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter


REPO_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = REPO_ROOT / "ft_normalized" / "Abeta"
MANIFEST = FT_DIR / "manifest.json"

VACCINE_DATASET = "mouse_app_vaccine_batch"

PCTL = [90.0, 95.0, 99.0, 99.5, 99.9]
SIZE_BINS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 1000), (1000, None)]
LOCAL_FILTER_SIZE = 15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    patch_id: str
    batch: str
    subject: str
    region: Optional[str]
    gold_path: Path

    # Distribution stats (norm and raw): {"p90": x, ..., "max": y, "max_over_p99": z}
    norm_stats: Dict[str, float] = field(default_factory=dict)
    raw_stats: Dict[str, float] = field(default_factory=dict)

    # Gold mask stats
    n_components: int = 0
    labeled_voxels: int = 0
    gold_fraction: float = 0.0

    # Per-component brightness (normalized space)
    comp_neg: int = 0            # mean_norm < 0
    comp_borderline: int = 0     # 0 <= mean_norm < 1
    comp_confident: int = 0      # mean_norm >= 1

    # Component size bins
    size_bins: Dict[str, int] = field(default_factory=dict)

    # Local brightness ratio (raw / local_mean_raw), averaged across components
    mean_local_ratio: float = 0.0

    # Flags
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def derive_patch_info(gold_path: Path) -> Tuple[str, str, str]:
    """From a path like ft_normalized/Abeta/<batch>/<subject>/<pid>_seg_gold.nii.gz
    return (patch_id, batch, subject).
    """
    parts = gold_path.parts
    # Walk up: subject = parent, batch = grandparent under Abeta
    subject = gold_path.parent.name
    batch = gold_path.parent.parent.name
    patch_id = gold_path.name.replace("_seg_gold.nii.gz", "")
    return patch_id, batch, subject


def discover_gold_files() -> Dict[str, Path]:
    """Collect every *_seg_gold.nii.gz under FT_DIR. Union of manifest +
    tree walk so nothing is missed.

    Returns {patch_id: gold_path}.
    """
    found: Dict[str, Path] = {}

    # From manifest (if present)
    if MANIFEST.exists():
        try:
            m = json.load(open(MANIFEST))
            for entry in m.get("patches", []):
                raw_path = entry.get("raw_path", "")
                if not raw_path:
                    continue
                pid = Path(raw_path).name.split("_crop")[0]
                # Gold file sits next to crop128
                cand = REPO_ROOT / Path(raw_path).parent / f"{pid}_seg_gold.nii.gz"
                if cand.exists():
                    found[pid] = cand
        except Exception as e:
            print(f"WARN: could not read manifest ({e}), falling back to tree walk only")

    # Tree walk (covers files not in manifest)
    for gp in sorted(FT_DIR.rglob("*_seg_gold.nii.gz")):
        if "wt_references" in gp.parts:
            continue
        pid = gp.name.replace("_seg_gold.nii.gz", "")
        if pid not in found:
            found[pid] = gp

    return found


# ---------------------------------------------------------------------------
# Per-patch computation
# ---------------------------------------------------------------------------

def _distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {f"p{p}": 0.0 for p in PCTL} | {"max": 0.0, "max_over_p99": 0.0}
    out: Dict[str, float] = {}
    for p in PCTL:
        out[f"p{p}"] = float(np.percentile(finite, p))
    out["max"] = float(finite.max())
    p99 = out["p99.0"]
    out["max_over_p99"] = float(out["max"] / p99) if p99 > 0 else float("inf")
    return out


def _size_bin_key(lo: int, hi: Optional[int]) -> str:
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


def _size_bins(sizes: np.ndarray) -> Dict[str, int]:
    bins = {_size_bin_key(lo, hi): 0 for lo, hi in SIZE_BINS}
    for s in sizes:
        s_int = int(s)
        for lo, hi in SIZE_BINS:
            if s_int >= lo and (hi is None or s_int < hi):
                bins[_size_bin_key(lo, hi)] += 1
                break
    return bins


def audit_one(patch_id: str, gold_path: Path) -> Optional[AuditResult]:
    batch, subject = gold_path.parent.parent.name, gold_path.parent.name
    patch_dir = gold_path.parent

    raw_path = patch_dir / f"{patch_id}_crop128.nii.gz"
    norm_path = patch_dir / f"{patch_id}_normalized128.nii.gz"
    meta_path = patch_dir / f"{patch_id}_meta.json"

    if not raw_path.exists() or not norm_path.exists():
        print(f"  WARN: {patch_id} missing raw/norm NIfTI, skipping")
        return None

    raw = nib.load(str(raw_path)).get_fdata().astype(np.float32)
    norm = nib.load(str(norm_path)).get_fdata().astype(np.float32)
    gold = nib.load(str(gold_path)).get_fdata().astype(np.uint8)
    gold_bin = (gold > 0).astype(np.uint8)

    region = None
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            region = meta.get("region_group")
        except Exception:
            pass

    res = AuditResult(
        patch_id=patch_id,
        batch=batch,
        subject=subject,
        region=region,
        gold_path=gold_path,
    )
    res.norm_stats = _distribution_stats(norm)
    res.raw_stats = _distribution_stats(raw)

    # Connected components (26-connectivity) on the gold mask
    labels = cc3d.connected_components(gold_bin, connectivity=26)
    n_total = int(labels.max())
    res.labeled_voxels = int(gold_bin.sum())
    res.gold_fraction = res.labeled_voxels / float(gold_bin.size)
    res.n_components = n_total

    if n_total == 0:
        # Empty mask — nothing to compute per-component
        res.size_bins = {_size_bin_key(lo, hi): 0 for lo, hi in SIZE_BINS}
        res.mean_local_ratio = 0.0
        return res

    # Per-component stats via np.bincount + labeled means.
    sizes = np.bincount(labels.ravel(), minlength=n_total + 1)
    sizes[0] = 0  # drop background
    comp_sizes = sizes[1:]  # per-component voxel counts

    # For per-component means in norm and raw space we sum values weighted
    # by component label. This is fast and avoids a Python loop.
    sum_norm = np.bincount(labels.ravel(), weights=norm.ravel(), minlength=n_total + 1)
    sum_raw = np.bincount(labels.ravel(), weights=raw.ravel(), minlength=n_total + 1)
    mean_norm_per_comp = sum_norm[1:] / np.maximum(comp_sizes, 1)
    mean_raw_per_comp = sum_raw[1:] / np.maximum(comp_sizes, 1)

    res.comp_neg = int((mean_norm_per_comp < 0).sum())
    res.comp_borderline = int(((mean_norm_per_comp >= 0) & (mean_norm_per_comp < 1)).sum())
    res.comp_confident = int((mean_norm_per_comp >= 1).sum())

    res.size_bins = _size_bins(comp_sizes)

    # Local brightness ratio: average of (mean_raw_component / mean_raw_local)
    # per component, weighted equally across components.
    local_mean = uniform_filter(raw, size=LOCAL_FILTER_SIZE)
    sum_local = np.bincount(labels.ravel(), weights=local_mean.ravel(), minlength=n_total + 1)
    mean_local_per_comp = sum_local[1:] / np.maximum(comp_sizes, 1)
    # Safe divide
    denom = np.where(mean_local_per_comp > 1e-6, mean_local_per_comp, np.nan)
    ratios = mean_raw_per_comp / denom
    valid = ratios[np.isfinite(ratios)]
    res.mean_local_ratio = float(valid.mean()) if valid.size else 0.0

    return res


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def apply_flags(r: AuditResult) -> None:
    if r.n_components == 0:
        # Empty mask → nothing to flag (a reviewer has explicitly said no plaques)
        return

    is_vaccine = r.batch == VACCINE_DATASET
    max_over_p99 = r.norm_stats.get("max_over_p99", 0.0)
    dim_frac = (r.comp_neg + r.comp_borderline) / max(r.n_components, 1)  # frac with mean_norm < 1
    neg_frac = r.comp_neg / max(r.n_components, 1)

    # HEAVY_TAIL_OVERSEG
    if (max_over_p99 > 10 and r.gold_fraction > 0.02 and dim_frac > 0.30):
        r.flags.append("HEAVY_TAIL_OVERSEG")

    # SPARSE_EXPECTED (vaccine only)
    if is_vaccine and max_over_p99 > 10 and r.gold_fraction > 0.015:
        r.flags.append("SPARSE_EXPECTED")

    # NEG_NORM_COMPONENTS
    if neg_frac > 0.20:
        r.flags.append("NEG_NORM_COMPONENTS")

    # TINY_DOMINATED: >50% of components are <10 voxels (sum of first two size bins)
    tiny = r.size_bins.get("0-5", 0) + r.size_bins.get("5-10", 0)
    if tiny / max(r.n_components, 1) > 0.50:
        r.flags.append("TINY_DOMINATED")

    # LOW_CONFIDENCE
    if r.mean_local_ratio < 1.10:
        r.flags.append("LOW_CONFIDENCE")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _fmt_stats(s: Dict[str, float]) -> str:
    return (
        f"p90={s['p90.0']:.2f}  p95={s['p95.0']:.2f}  p99={s['p99.0']:.2f}  "
        f"p99.5={s['p99.5']:.2f}  p99.9={s['p99.9']:.2f}  max={s['max']:.2f}  "
        f"max/p99={s['max_over_p99']:.2f}"
    )


def _fmt_size_bins(bins: Dict[str, int]) -> str:
    keys_in_order = [_size_bin_key(lo, hi) for lo, hi in SIZE_BINS]
    return "  ".join(f"{k}:{bins.get(k, 0)}" for k in keys_in_order)


def write_report(results: List[AuditResult], out_path: Path) -> None:
    # Sort: more flags first, then alphabetical
    results_sorted = sorted(results, key=lambda r: (-len(r.flags), r.patch_id))

    # Summary counts
    n_total = len(results)
    n_flagged = sum(1 for r in results if r.flags)
    flag_counts: Dict[str, int] = {}
    for r in results:
        for f in r.flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("GOLD PATCH AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"scanned {FT_DIR}")
    lines.append(f"patches audited:   {n_total}")
    lines.append(f"with flag(s):      {n_flagged}  ({_fmt_pct(n_flagged/max(n_total,1))})")
    lines.append(f"clean:             {n_total - n_flagged}")
    lines.append("")
    lines.append("per-flag counts:")
    for fname in sorted(flag_counts):
        lines.append(f"  {fname:<24} {flag_counts[fname]}")
    if not flag_counts:
        lines.append("  (none)")
    lines.append("")

    # Top-10 most-suspect
    lines.append("top 10 most-suspect (by flag count):")
    top = [r for r in results_sorted if r.flags][:10]
    if not top:
        lines.append("  (none flagged)")
    else:
        for r in top:
            lines.append(
                f"  {len(r.flags)} flag(s)  {r.batch}/{r.subject}/{r.patch_id:<28} "
                f"[{','.join(r.flags)}]"
            )
    lines.append("")

    # Per-patch sections
    lines.append("=" * 80)
    lines.append("PER-PATCH DETAILS  (flagged first, then clean)")
    lines.append("=" * 80)
    for r in results_sorted:
        vaccine_note = (
            "  (vaccine batch — WT pooled cross-batch, expect noisier norm)"
            if r.batch == VACCINE_DATASET else ""
        )
        lines.append("")
        lines.append("-" * 80)
        tag = f"[{','.join(r.flags)}]" if r.flags else "[clean]"
        lines.append(f"{r.patch_id}   {tag}")
        lines.append(f"  batch:    {r.batch}{vaccine_note}")
        lines.append(f"  subject:  {r.subject}")
        lines.append(f"  region:   {r.region or '?'}")
        lines.append(f"  gold:     {r.gold_path}")
        lines.append(f"")
        lines.append(f"  norm  {_fmt_stats(r.norm_stats)}")
        lines.append(f"  raw   {_fmt_stats(r.raw_stats)}")
        lines.append(f"")
        lines.append(
            f"  gold mask: components={r.n_components}  voxels={r.labeled_voxels}  "
            f"fraction={_fmt_pct(r.gold_fraction)}"
        )
        lines.append(
            f"  per-component mean_norm: neg(<0)={r.comp_neg}  "
            f"borderline(0..1)={r.comp_borderline}  confident(>=1)={r.comp_confident}"
        )
        lines.append(f"  size bins (voxels): {_fmt_size_bins(r.size_bins)}")
        lines.append(
            f"  mean local brightness ratio (raw / uniform_filter{LOCAL_FILTER_SIZE}): "
            f"{r.mean_local_ratio:.3f}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out", default=str(FT_DIR / "gold_audit.txt"),
        help="output report path (default: ft_normalized/Abeta/gold_audit.txt)",
    )
    args = p.parse_args()

    gold_files = discover_gold_files()
    if not gold_files:
        sys.exit(f"no *_seg_gold.nii.gz found under {FT_DIR}")

    print(f"auditing {len(gold_files)} gold patches ...")
    results: List[AuditResult] = []
    for pid in sorted(gold_files):
        gp = gold_files[pid]
        try:
            r = audit_one(pid, gp)
            if r is None:
                continue
            apply_flags(r)
            results.append(r)
            tag = f"[{','.join(r.flags)}]" if r.flags else "[clean]"
            print(f"  {pid:<28} {tag}")
        except Exception as e:
            print(f"  FAIL {pid}: {type(e).__name__}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, out_path)

    n_flagged = sum(1 for r in results if r.flags)
    print(f"\nwrote {out_path}")
    print(f"  total: {len(results)}  flagged: {n_flagged}  clean: {len(results)-n_flagged}")


if __name__ == "__main__":
    main()
