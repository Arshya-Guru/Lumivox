"""Binarize a uint8 probability-map .ozx into a uint8 mask .ome.zarr (directory).

Reads a previously-produced probability-map .ozx (zipped OME-NGFF v0.5, uint8
0-255 = sigmoid * 255), thresholds it, and writes a separate multi-resolution
OME-Zarr v3 directory at the same target location with the binary mask.

The output mirrors spimquant's ozx layout (4D [c, z, y, x], chunks 1×128³,
scale in mm, unit=millimeter, xyz_orientation at root attrs).

Usage:
    python scripts/binarize_ozx_to_omezarr.py \\
        --input  path/to/...probmap_fixed.ozx \\
        --output path/to/...mask.ome.zarr \\
        --threshold 0.5 \\
        --orientation IAR
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore, LocalStore
from zarr.codecs import BloscCodec


OUT_CHUNK = 128


def downsample_uint8_yx_4d(level0: zarr.Array, factor: int) -> np.ndarray:
    """Downsample 4D [c, z, y, x] uint8 by `factor` along y and x.

    Uses MAX pooling — appropriate for binary masks where the source values are
    {0, 1}. Plaques are sparse and small; mean-pool + threshold would erase
    almost all foreground at coarse pyramid levels, leaving the mask invisible
    at low zoom in napari. Max-pool keeps any foreground voxel visible at every
    pyramid level (over-represents area slightly but that's what you want for
    sparse-target viewing).
    """
    if level0.ndim == 4:
        C, Z, Y, X = level0.shape
    elif level0.ndim == 3:
        C = 1
        Z, Y, X = level0.shape
    else:
        raise ValueError(f"Unexpected level0 ndim={level0.ndim}, shape={level0.shape}")
    Yd, Xd = Y // factor, X // factor
    out = np.zeros((C, Z, Yd, Xd), dtype=np.uint8)
    z_step = max(1, min(128, Z))
    for c in range(C):
        for z0 in range(0, Z, z_step):
            z1 = min(z0 + z_step, Z)
            if level0.ndim == 4:
                block = level0[c, z0:z1, : Yd * factor, : Xd * factor]
            else:
                block = level0[z0:z1, : Yd * factor, : Xd * factor]
            # Reshape (Z, Y, X) -> (Z, Yd, factor, Xd, factor) and max over the
            # two factor axes.
            block = block.reshape(z1 - z0, Yd, factor, Xd, factor).max(axis=(2, 4))
            out[c, z0:z1] = block.astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Input .ozx (zipped OME-NGFF v0.5 probability map).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output .ome.zarr directory (will be created/overwritten).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold in [0, 1]. Default 0.5.")
    ap.add_argument("--orientation", default="IAR",
                    help="xyz_orientation to declare (matches AS36F4 input by default).")
    ap.add_argument("--build-pyramid", type=int, default=2,
                    help="Number of additional yx-downsampled levels (0 = level 0 only).")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    thr_uint8 = int(round(args.threshold * 255))
    print(f"Binarizing {args.input}  thr={args.threshold} -> uint8>{thr_uint8}")

    # ----- Open source ozx -----
    src_store = ZipStore(str(args.input), mode="r")
    src_root = zarr.open_group(store=src_store, mode="r")
    src_lvl0 = src_root["0"]
    print(f"  source level 0 shape={src_lvl0.shape} dtype={src_lvl0.dtype}")

    # ----- Set up destination ome.zarr -----
    if args.output.exists():
        print(f"  removing existing output: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=False)

    dst_store = LocalStore(str(args.output))
    dst_root = zarr.create_group(store=dst_store, zarr_format=3, overwrite=True)

    # 4D shape [c=1, z, y, x] — matches spimquant.
    if src_lvl0.ndim == 4:
        _, Z, Y, X = src_lvl0.shape
    else:
        Z, Y, X = src_lvl0.shape

    out0 = dst_root.create_array(
        name="0",
        shape=(1, Z, Y, X),
        chunks=(1, OUT_CHUNK, OUT_CHUNK, OUT_CHUNK),
        dtype="uint8",
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")],
        fill_value=0,
        dimension_names=["c", "z", "y", "x"],
    )

    # ----- Stream binarize level 0 -----
    print(f"  binarizing level 0 ({Z}×{Y}×{X}) in z-slabs of 128...")
    z_step = 128
    n_fg = 0
    for z0 in range(0, Z, z_step):
        z1 = min(z0 + z_step, Z)
        if src_lvl0.ndim == 4:
            slab = src_lvl0[0, z0:z1, :, :]
        else:
            slab = src_lvl0[z0:z1, :, :]
        mask = (slab > thr_uint8).astype(np.uint8)
        n_fg += int(mask.sum())
        out0[0, z0:z1, :, :] = mask
    total_voxels = Z * Y * X
    print(f"  level 0 foreground: {n_fg} / {total_voxels} ({100*n_fg/total_voxels:.4f}%)")

    # ----- Read scale + orientation from source -----
    # zarr v3's ZipStore.get() is async (coroutine), so go through src_root.attrs.
    src_attr_dict = dict(src_root.attrs)
    src_ome = src_attr_dict.get("ome", {})
    src_multiscales = src_ome.get("multiscales", [{}])
    src_datasets = src_multiscales[0].get("datasets", [])
    if not src_datasets:
        raise SystemExit("Source ozx has no multiscales/datasets metadata — was it patched?")
    src_scale = src_datasets[0]["coordinateTransformations"][0]["scale"]
    # Source may be 3-element (zyx) or 4-element (czyx) — normalize to (z, y, x) tuple.
    if len(src_scale) == 4:
        z_mm, y_mm, x_mm = float(src_scale[1]), float(src_scale[2]), float(src_scale[3])
    elif len(src_scale) == 3:
        z_mm, y_mm, x_mm = float(src_scale[0]), float(src_scale[1]), float(src_scale[2])
    else:
        raise SystemExit(f"Unexpected scale length: {src_scale}")
    print(f"  scale (mm): z={z_mm} y={y_mm} x={x_mm}")

    # ----- Build pyramid -----
    pyramid_paths = ["0"]
    pyramid_scales = [(z_mm, y_mm, x_mm)]
    for ilev in range(1, args.build_pyramid + 1):
        factor = 2 ** ilev
        print(f"  building level {ilev} (yx /{factor})...")
        downsampled = downsample_uint8_yx_4d(out0, factor)
        outL = dst_root.create_array(
            name=str(ilev),
            shape=downsampled.shape,
            chunks=(1, OUT_CHUNK, OUT_CHUNK, OUT_CHUNK),
            dtype="uint8",
            compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")],
            fill_value=0,
            dimension_names=["c", "z", "y", "x"],
        )
        outL[:] = downsampled
        pyramid_paths.append(str(ilev))
        pyramid_scales.append((z_mm, y_mm * factor, x_mm * factor))

    # ----- Write OME-NGFF v0.5 metadata + provenance -----
    dst_root_zj = args.output / "zarr.json"
    rm = json.loads(dst_root_zj.read_text())
    rm.setdefault("attributes", {})
    rm["attributes"]["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": "/",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "millimeter"},
                {"name": "y", "type": "space", "unit": "millimeter"},
                {"name": "x", "type": "space", "unit": "millimeter"},
            ],
            "datasets": [
                {
                    "path": path,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, sc[0], sc[1], sc[2]]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0]},
                    ],
                }
                for path, sc in zip(pyramid_paths, pyramid_scales)
            ],
        }],
    }
    if args.orientation:
        rm["attributes"]["xyz_orientation"] = args.orientation
    rm["attributes"]["_lumivox_binarize"] = {
        "source": str(args.input),
        "threshold": args.threshold,
        "threshold_uint8": thr_uint8,
        "orientation_declared": args.orientation,
    }
    dst_root_zj.write_text(json.dumps(rm, indent=2))

    size_gb = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) / 1e9
    print(f"Done: {args.output}  ({size_gb:.3f} GB)")


if __name__ == "__main__":
    main()
