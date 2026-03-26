"""Extract patches from OME-Zarr manifests and save as .npy files.

Borrows the same extraction logic as OMEZarrPatchDataset._load_sample,
minus augmentations and training. Single process, sequential, restartable.

Usage:
    pixi run python scripts/cache_patches.py \
        --manifest manifests/abeta_50k_alldata.json \
        --output-dir /nfs/scratch/apooladi/training_patches/abeta
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from zarrnii import ZarrNii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    patches = manifest["patches"]
    cfg = manifest["config"]
    patch_size = cfg["patch_size"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifest for provenance
    manifest_dest = output_dir / "manifest.json"
    if not manifest_dest.exists():
        shutil.copy2(args.manifest, manifest_dest)

    # Group by zarr volume so we open each once
    groups = defaultdict(list)
    for idx, entry in enumerate(patches):
        key = (entry["zarr_path"], entry.get("stain_channel", 0), entry.get("zarr_source", "resampled"))
        groups[key].append((idx, entry))

    n_exist = sum(1 for i in range(len(patches)) if (output_dir / f"patch_{i:06d}.npy").exists())
    print(f"{len(patches)} patches, {len(groups)} volumes, {n_exist} already cached")

    if n_exist == len(patches):
        print("All done.")
        return

    t0 = time.time()
    saved = 0
    skipped = 0
    failed = 0
    half = patch_size // 2

    for gi, ((zarr_path, stain_channel, zarr_source), entries) in enumerate(groups.items(), 1):
        subj = Path(zarr_path).parent.name
        print(f"[{gi}/{len(groups)}] {subj} — {len(entries)} patches, opening zarr...", flush=True)

        try:
            kwargs = dict(channels=[stain_channel])
            if zarr_source == "fullres":
                kwargs["downsample_near_isotropic"] = True
            znimg = ZarrNii.from_ome_zarr(zarr_path, **kwargs)
            darr = znimg.darr
        except Exception as e:
            print(f"  SKIP volume: {e}", flush=True)
            failed += len(entries)
            continue

        for pi, (idx, entry) in enumerate(entries, 1):
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
                    patch_zn = znimg.crop_centered(center, patch_size=(patch_size, patch_size, patch_size))
                    vol = patch_zn.darr.compute().astype(np.float32)

                vol_u16 = np.clip(vol, 0, 65535).astype(np.uint16)
                np.save(str(out_path), vol_u16)
                saved += 1

                if saved % 100 == 0:
                    elapsed = time.time() - t0
                    total_done = saved + skipped
                    print(f"  {subj}: {pi}/{len(entries)} | total saved={saved}/{len(patches)} ({elapsed:.0f}s)", flush=True)

            except Exception as e:
                failed += 1
                print(f"  [FAIL] patch {idx}: {e}", flush=True)

        print(f"  done — saved={saved} skipped={skipped} failed={failed}", flush=True)
        del znimg, darr

    elapsed = time.time() - t0
    n_final = sum(1 for i in range(len(patches)) if (output_dir / f"patch_{i:06d}.npy").exists())
    print(f"\nFinished in {elapsed:.0f}s — {n_final}/{len(patches)} cached (failed={failed})")


if __name__ == "__main__":
    main()
