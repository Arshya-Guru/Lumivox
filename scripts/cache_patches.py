"""Extract patches from OME-Zarr manifests and save as .npy files on disk.

Decouples the slow NFS → local extraction step from training.  Patches are
grouped by source zarr volume so each volume is only opened once.  The script
is restartable — existing .npy files are skipped.

Usage:
    pixi run python scripts/cache_patches.py \
        --manifest manifests/abeta_50k_alldata.json \
        --output-dir /nfs/scratch/apooladi/training_patches/abeta \
        --workers 8

    pixi run python scripts/cache_patches.py \
        --manifest manifests/iba1_50k_ki3.json \
        --output-dir /nfs/scratch/apooladi/training_patches/iba1 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_patches_for_volume(
    zarr_path: str,
    stain_channel: int,
    zarr_source: str,
    patch_entries: list[tuple[int, dict]],
    patch_size: int,
    output_dir: Path,
) -> tuple[int, int, int]:
    """Extract all patches for one zarr volume. Returns (saved, skipped, failed)."""
    from zarrnii import ZarrNii

    subj_name = Path(zarr_path).parent.name
    print(f"  Opening zarr: {subj_name} ...", flush=True)

    kwargs = dict(channels=[stain_channel])
    if zarr_source == "fullres":
        kwargs["downsample_near_isotropic"] = True

    znimg = ZarrNii.from_ome_zarr(zarr_path, **kwargs)
    darr = znimg.darr
    print(f"  Zarr open: {subj_name}, shape={darr.shape}", flush=True)

    saved, skipped, failed = 0, 0, 0
    half = patch_size // 2
    total = len(patch_entries)
    subj = Path(zarr_path).parent.name

    for pi, (idx, entry) in enumerate(patch_entries, 1):
        out_path = output_dir / f"patch_{idx:06d}.npy"
        if out_path.exists():
            skipped += 1
            continue

        try:
            if "center_vox" in entry:
                cz = int(round(entry["center_vox"][0]))
                cy = int(round(entry["center_vox"][1]))
                cx = int(round(entry["center_vox"][2]))

                z0, z1 = max(0, cz - half), min(darr.shape[-3], cz + half)
                y0, y1 = max(0, cy - half), min(darr.shape[-2], cy + half)
                x0, x1 = max(0, cx - half), min(darr.shape[-1], cx + half)

                vol = darr[0, z0:z1, y0:y1, x0:x1].compute().astype(np.float32)
            else:
                center = tuple(entry["center_phys"])
                patch_zn = znimg.crop_centered(
                    center, patch_size=(patch_size, patch_size, patch_size)
                )
                vol = patch_zn.darr.compute().astype(np.float32)

            vol_u16 = np.clip(vol, 0, 65535).astype(np.uint16)
            # Write to temp file then rename for atomicity
            tmp_path = out_path.with_suffix(".npy.tmp")
            np.save(tmp_path, vol_u16)
            tmp_path.rename(out_path)
            saved += 1
            if saved % 50 == 0 or pi == total:
                done = saved + skipped + failed
                print(f"    {subj}: {done}/{total} patches ({saved} saved)", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [FAIL] patch {idx}: {e}", flush=True)

    return saved, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract patches from a manifest and save as .npy files"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy files")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each handles one zarr volume at a time)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    patches = manifest["patches"]
    cfg = manifest["config"]
    patch_size = cfg["patch_size"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifest alongside the patches for provenance
    manifest_dest = output_dir / "manifest.json"
    if not manifest_dest.exists():
        shutil.copy2(args.manifest, manifest_dest)
        print(f"Copied manifest → {manifest_dest}")

    # Group patches by (zarr_path, stain_channel, zarr_source)
    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for idx, entry in enumerate(patches):
        key = (
            entry["zarr_path"],
            entry.get("stain_channel", 0),
            entry.get("zarr_source", "resampled"),
        )
        groups[key].append((idx, entry))

    # Count already-cached
    n_existing = sum(1 for i in range(len(patches)) if (output_dir / f"patch_{i:06d}.npy").exists())
    print(
        f"Manifest: {len(patches)} patches, {len(groups)} volumes, "
        f"stain={cfg['stain']}, patch_size={patch_size}"
    )
    print(f"Output:   {output_dir}")
    print(f"Already cached: {n_existing}/{len(patches)}")
    if n_existing == len(patches):
        print("All patches already cached — nothing to do.")
        return

    t0 = time.time()
    total_saved, total_skipped, total_failed = 0, 0, 0

    if args.workers <= 1:
        # Single-process: iterate volume groups sequentially
        for i, ((zpath, ch, zsrc), entries) in enumerate(groups.items(), 1):
            subj = Path(zpath).parent.name
            print(
                f"[{i}/{len(groups)}] {subj} ({len(entries)} patches) ...",
                end="",
                flush=True,
            )
            saved, skipped, failed = extract_patches_for_volume(
                zpath, ch, zsrc, entries, patch_size, output_dir
            )
            total_saved += saved
            total_skipped += skipped
            total_failed += failed
            print(f"  saved={saved} skipped={skipped} failed={failed}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for (zpath, ch, zsrc), entries in groups.items():
                fut = pool.submit(
                    extract_patches_for_volume,
                    zpath, ch, zsrc, entries, patch_size, output_dir,
                )
                futures[fut] = (zpath, len(entries))

            for i, fut in enumerate(as_completed(futures), 1):
                zpath, n = futures[fut]
                subj = Path(zpath).parent.name
                try:
                    saved, skipped, failed = fut.result()
                    total_saved += saved
                    total_skipped += skipped
                    total_failed += failed
                    print(
                        f"[{i}/{len(groups)}] {subj} ({n} patches): "
                        f"saved={saved} skipped={skipped} failed={failed}"
                    )
                except Exception as e:
                    total_failed += n
                    print(f"[{i}/{len(groups)}] {subj}: CRASHED — {e}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s — "
        f"saved={total_saved}, skipped={total_skipped}, failed={total_failed}"
    )
    n_total = sum(1 for i in range(len(patches)) if (output_dir / f"patch_{i:06d}.npy").exists())
    print(f"Total cached: {n_total}/{len(patches)}")


if __name__ == "__main__":
    main()
