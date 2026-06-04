"""Whole-brain sliding-window inference for FT segmentation.

Reads channel <c> at resolution level <L> of an OME-Zarr v2 input volume,
slides a 128^3 window over channel <c> with stride <S> (default 64 = 50%
overlap), per-tile z-scores using only real (non-padded) voxels, forwards
through the FT segmentation model, sigmoids, and blends the resulting
probabilities into uint8 (0-255) via Gaussian weighting. Result is written as
multi-resolution OME-NGFF v0.5 zarr v3 packed as an .ozx file (zip, store).

Processing is super-tile based to bound memory. With patch=128, stride=64,
super-tile inner=384, halo=64, buffer=512, in-RAM accumulator per super-tile
is ~3 GB. Output is built one super-tile at a time.

For stride=128 the script degenerates to v1's non-overlapping behavior
(no Gaussian blending, single forward per output voxel) but still uses the
super-tile loop.

Usage:
    python scripts/infer_finetune_brain.py \\
        --input-zarr .../sub-AS36F4_..._SPIM.ome.zarr \\
        --channel 0 --level 0 \\
        --checkpoint checkpoints/finetune_abeta_nnbyol3d_frozen/finetune-ep0064-dice0.7350.ckpt \\
        --output-ozx .../sub-AS36F4_..._desc-lumivoxFT...nnbyol3d..._probmap.ozx \\
        --stride 64 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Make `from lumivox.X import Y` work when this script is run by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec

from lumivox.training.finetune import build_segmentation_model


PATCH = 128
OUT_CHUNK = 128
DEFAULT_SUPER_INNER = 384       # super-tile inner edge length (writes contribute here)
DEFAULT_SUPER_INNER_NO_OVERLAP = 512  # bigger when no halo needed
MIN_VALID_VOXELS_FOR_STATS = 4096     # require at least this many real voxels to z-score


def load_lightning_state_dict(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def gaussian_weight_kernel(patch: int, sigma: float) -> np.ndarray:
    """3D Gaussian centered at the patch center, normalized so max == 1.

    Matches the nnU-Net convention: sigma = patch / 8 gives a noticeable but
    not extreme falloff (corners at ~e^(-32) ≈ 0, edges at center axes at
    ~e^(-2) ≈ 0.135). nnU-Net's default is sigma = patch * 0.125 = 16 for 128.
    """
    coords = np.arange(patch, dtype=np.float32) - (patch - 1) / 2.0
    g1 = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    kernel /= kernel.max()
    kernel = kernel.astype(np.float32)
    # Floor to avoid pure-zero contributions at corners; downstream weight_acc
    # accumulates these and we want non-zero "I've seen this voxel" signals.
    kernel = np.maximum(kernel, 1e-3)
    return kernel


def downsample_uint8_yx(level0: zarr.Array, factor: int) -> np.ndarray:
    """Downsample a 4D [c, z, y, x] uint8 zarr by `factor` along y and x.
    Returns a 4D numpy array."""
    C, Z, Y, X = level0.shape
    Yd, Xd = Y // factor, X // factor
    out = np.zeros((C, Z, Yd, Xd), dtype=np.uint8)
    z_step = max(1, min(128, Z))
    for c in range(C):
        for z0 in range(0, Z, z_step):
            z1 = min(z0 + z_step, Z)
            block = level0[c, z0:z1, : Yd * factor, : Xd * factor].astype(np.float32)
            block = block.reshape(z1 - z0, Yd, factor, Xd, factor).mean(axis=(2, 4))
            out[c, z0:z1] = np.clip(block + 0.5, 0, 255).astype(np.uint8)
    return out


def read_input_zattrs(input_zarr: Path) -> dict:
    """Load the input zarr's root .zattrs (v0.4) so we can mirror its conventions."""
    return json.loads((input_zarr / ".zattrs").read_text())


def read_input_scale(input_zattrs: dict, level: int) -> tuple:
    """Return (z, y, x) scale at the given level, in whatever units the input uses.

    NOTE: the input file declares unit=micrometer but the values are actually in
    millimeters (0.004 = 4 um = 0.004 mm). zarrnii and other tools appear to
    treat the scale value as millimeter regardless of the declared unit, so we
    pass through the value unchanged and declare the unit as 'millimeter' on
    our output — matching spimquant's ozx convention.
    """
    ds = input_zattrs["multiscales"][0]["datasets"][level]
    scale = ds["coordinateTransformations"][0]["scale"]
    return scale[1], scale[2], scale[3]


def write_ome_ngff_v05(root_path: Path, levels: list, scales_mm: list,
                       xyz_orientation: str | None = None,
                       name: str = "/") -> None:
    """Write OME-NGFF v0.5 multiscale metadata for a 4D [c, z, y, x] output.

    Uses millimeter unit + values directly in mm (no 1000x conversion).
    Adds xyz_orientation as a top-level attribute (sibling of `ome`) so that
    zarrnii and similar tools pick up the correct anatomical orientation.
    """
    root_zj = root_path / "zarr.json"
    root_meta = json.loads(root_zj.read_text())
    root_meta["attributes"] = root_meta.get("attributes", {})
    root_meta["attributes"]["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": name,
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
                        {"type": "scale", "scale": [
                            1.0,             # channel
                            float(sc[0]),    # z (mm)
                            float(sc[1]),    # y (mm)
                            float(sc[2]),    # x (mm)
                        ]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0]},
                    ],
                }
                for path, sc in zip(levels, scales_mm)
            ],
        }],
    }
    if xyz_orientation:
        root_meta["attributes"]["xyz_orientation"] = xyz_orientation
    root_zj.write_text(json.dumps(root_meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-zarr", required=True, type=Path)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-ozx", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=64,
                    help="Sliding-window stride. Must divide PATCH (128). "
                         "64 = 50%% overlap (default, with Gaussian blending). "
                         "128 = no overlap (v1 behavior, no blending).")
    ap.add_argument("--gaussian-sigma", type=float, default=PATCH / 8.0,
                    help="Sigma for the Gaussian weight kernel (in voxels). Default patch/8 = 16.")
    ap.add_argument("--super-inner", type=int, default=0,
                    help="Super-tile inner edge (per axis). 0 = auto-pick based on stride.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Window batch size per forward. Default 32 — observed peak "
                         "VRAM <20 GB on L40S/A100 at this size during the first "
                         "v2 runs, so this is the safe default for those GPUs.")
    ap.add_argument("--build-pyramid", type=int, default=2,
                    help="Number of additional yx-downsampled levels (0 = level 0 only).")
    ap.add_argument("--limit-supertiles", type=int, default=0,
                    help="If >0, stop after this many super-tiles (smoke test).")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Temp dir for output zarr (defaults to system tmp).")
    args = ap.parse_args()

    args.output_ozx.parent.mkdir(parents=True, exist_ok=True)

    if PATCH % args.stride != 0:
        raise SystemExit(
            f"--stride must divide PATCH={PATCH}. Got stride={args.stride}."
        )
    halo = max(0, PATCH - args.stride)  # 0 when stride==patch
    use_blend = halo > 0
    if args.super_inner > 0:
        inner = args.super_inner
    else:
        inner = DEFAULT_SUPER_INNER if use_blend else DEFAULT_SUPER_INNER_NO_OVERLAP
    if inner % args.stride != 0:
        # Round up to a multiple of stride to keep window positions aligned.
        inner = math.ceil(inner / args.stride) * args.stride
    buf_size = inner + 2 * halo

    # ----- Open input -----
    inroot = zarr.open(str(args.input_zarr), mode="r")
    arr_in = inroot[str(args.level)]
    C, Z, Y, X = arr_in.shape
    input_zattrs = read_input_zattrs(args.input_zarr)
    z_mm, y_mm, x_mm = read_input_scale(input_zattrs, args.level)
    input_orientation = input_zattrs.get("xyz_orientation")  # may be None
    print(f"Input zarr:    {args.input_zarr}")
    print(f"  level {args.level} shape: {arr_in.shape}  dtype: {arr_in.dtype}  chunks: {arr_in.chunks}")
    print(f"  voxel size (mm): z={z_mm} y={y_mm} x={x_mm}")
    print(f"  xyz_orientation: {input_orientation or '(absent — output will not declare one)'}")
    print(f"  using channel {args.channel}")
    print(f"Sliding config:")
    print(f"  patch={PATCH}  stride={args.stride}  halo={halo}  "
          f"super-tile inner={inner}  buffer={buf_size}  blend={'yes' if use_blend else 'no'}")
    if use_blend:
        print(f"  gaussian sigma={args.gaussian_sigma}")

    # ----- Load model -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model device: {device}")
    print(f"Checkpoint:   {args.checkpoint}")
    model = build_segmentation_model(
        checkpoint_path=None, num_classes=1, deep_supervision=True, dropout=0.0
    )
    sd = load_lightning_state_dict(str(args.checkpoint))
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  state dict: {len(sd)} keys loaded  missing={len(miss)}  unexpected={len(unexp)}")
    model.eval().to(device)

    # ----- Precompute Gaussian weight kernel -----
    if use_blend:
        weight_kernel = gaussian_weight_kernel(PATCH, args.gaussian_sigma)
    else:
        weight_kernel = np.ones((PATCH, PATCH, PATCH), dtype=np.float32)

    # ----- Set up output zarr v3 -----
    tmp_root = tempfile.mkdtemp(prefix="lumivox_infer_", dir=args.tmp_dir)
    out_zarr_dir = Path(tmp_root) / "out.zarr"
    print(f"Temp output zarr: {out_zarr_dir}")

    store = LocalStore(str(out_zarr_dir))
    root = zarr.create_group(store=store, zarr_format=3, overwrite=True)
    # 4D shape [c=1, z, y, x] with dimension_names — matches spimquant ozx layout.
    out0 = root.create_array(
        name="0",
        shape=(1, Z, Y, X),
        chunks=(1, OUT_CHUNK, OUT_CHUNK, OUT_CHUNK),
        dtype="uint8",
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")],
        fill_value=0,
        dimension_names=["c", "z", "y", "x"],
    )

    # ----- Super-tile loop -----
    n_supertiles_total = (
        ((Z + inner - 1) // inner)
        * ((Y + inner - 1) // inner)
        * ((X + inner - 1) // inner)
    )
    print(f"Super-tiles to process: {n_supertiles_total}")
    if args.limit_supertiles:
        print(f"Limit-supertiles enabled — stopping after {args.limit_supertiles}")

    n_supertiles_done = 0
    n_windows_done = 0
    t0 = time.time()
    t_last = t0

    for sz0 in range(0, Z, inner):
        for sy0 in range(0, Y, inner):
            for sx0 in range(0, X, inner):
                if args.limit_supertiles and n_supertiles_done >= args.limit_supertiles:
                    break

                # Brain coords of inner region this super-tile owns:
                inner_z1 = min(Z, sz0 + inner)
                inner_y1 = min(Y, sy0 + inner)
                inner_x1 = min(X, sx0 + inner)
                inner_dz = inner_z1 - sz0
                inner_dy = inner_y1 - sy0
                inner_dx = inner_x1 - sx0

                # Build buffer (size buf_size in each axis): we use the full buf_size
                # even for partial super-tiles, with zero-pad outside brain bounds.
                src_z0, src_z1 = sz0 - halo, sz0 + inner + halo
                src_y0, src_y1 = sy0 - halo, sy0 + inner + halo
                src_x0, src_x1 = sx0 - halo, sx0 + inner + halo
                rz0 = max(0, src_z0); rz1 = min(Z, src_z1)
                ry0 = max(0, src_y0); ry1 = min(Y, src_y1)
                rx0 = max(0, src_x0); rx1 = min(X, src_x1)

                buf = np.zeros((buf_size, buf_size, buf_size), dtype=np.float32)
                valid = np.zeros((buf_size, buf_size, buf_size), dtype=bool)
                if rz1 > rz0 and ry1 > ry0 and rx1 > rx0:
                    bz0, by0, bx0 = rz0 - src_z0, ry0 - src_y0, rx0 - src_x0
                    bz1, by1, bx1 = bz0 + (rz1 - rz0), by0 + (ry1 - ry0), bx0 + (rx1 - rx0)
                    buf[bz0:bz1, by0:by1, bx0:bx1] = arr_in[args.channel, rz0:rz1, ry0:ry1, rx0:rx1].astype(np.float32)
                    valid[bz0:bz1, by0:by1, bx0:bx1] = True

                # Accumulators
                prob_acc = np.zeros((buf_size, buf_size, buf_size), dtype=np.float32)
                weight_acc = np.zeros((buf_size, buf_size, buf_size), dtype=np.float32)

                # Slide windows inside the buffer
                window_positions = []
                for pz in range(0, buf_size - PATCH + 1, args.stride):
                    for py in range(0, buf_size - PATCH + 1, args.stride):
                        for px in range(0, buf_size - PATCH + 1, args.stride):
                            window_positions.append((pz, py, px))

                batch_imgs, batch_pos = [], []

                def flush():
                    nonlocal n_windows_done
                    if not batch_imgs:
                        return
                    x = torch.from_numpy(np.stack(batch_imgs)).unsqueeze(1).to(device, non_blocking=True)
                    with torch.no_grad():
                        if device == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                y = model(x)
                        else:
                            y = model(x)
                        if isinstance(y, (list, tuple)):
                            y = y[0]
                        probs = torch.sigmoid(y.float())[:, 0].cpu().numpy()
                    for (pz, py, px), p in zip(batch_pos, probs):
                        # Weighted contribution
                        prob_acc[pz:pz+PATCH, py:py+PATCH, px:px+PATCH] += p * weight_kernel
                        weight_acc[pz:pz+PATCH, py:py+PATCH, px:px+PATCH] += weight_kernel
                    n_windows_done += len(batch_imgs)
                    batch_imgs.clear()
                    batch_pos.clear()

                for pz, py, px in window_positions:
                    window = buf[pz:pz+PATCH, py:py+PATCH, px:px+PATCH]
                    window_valid = valid[pz:pz+PATCH, py:py+PATCH, px:px+PATCH]
                    n_valid = int(window_valid.sum())
                    if n_valid < MIN_VALID_VOXELS_FOR_STATS:
                        # Window is mostly outside brain — skip (it contributes 0 to acc,
                        # which is fine because at output we divide by weight_acc).
                        continue
                    # Z-score using only valid voxels
                    real = window[window_valid]
                    mu = float(real.mean())
                    sd = float(real.std())
                    if sd > 1e-8:
                        win_z = (window - mu) / sd
                    else:
                        win_z = window - mu
                    # Pad pixels: set z-scored value to 0 (neutral)
                    win_z = np.where(window_valid, win_z, 0.0).astype(np.float32)
                    batch_imgs.append(win_z)
                    batch_pos.append((pz, py, px))
                    if len(batch_imgs) >= args.batch_size:
                        flush()
                flush()

                # Compute final probability (only for inner region, and only valid voxels)
                inner_prob = prob_acc[halo:halo+inner_dz, halo:halo+inner_dy, halo:halo+inner_dx]
                inner_weight = weight_acc[halo:halo+inner_dz, halo:halo+inner_dy, halo:halo+inner_dx]
                inner_valid = valid[halo:halo+inner_dz, halo:halo+inner_dy, halo:halo+inner_dx]
                final = np.where(
                    inner_weight > 1e-6,
                    inner_prob / np.maximum(inner_weight, 1e-6),
                    0.0,
                )
                # Zero out positions that fell entirely outside brain (shouldn't happen
                # for the inner region but be defensive).
                final = np.where(inner_valid, final, 0.0)
                final = np.clip(final * 255.0 + 0.5, 0, 255).astype(np.uint8)
                # out0 is 4D [c=1, z, y, x] — write into the single channel slot.
                out0[0, sz0:inner_z1, sy0:inner_y1, sx0:inner_x1] = final

                n_supertiles_done += 1
                if n_supertiles_done % 5 == 0 or n_supertiles_done == n_supertiles_total:
                    now = time.time()
                    rate = 5.0 / (now - t_last) if now > t_last else 0.0
                    eta_s = (n_supertiles_total - n_supertiles_done) / max(rate, 1e-3)
                    print(
                        f"  supertile {n_supertiles_done}/{n_supertiles_total}  "
                        f"pos=(z={sz0},y={sy0},x={sx0})  "
                        f"win/sup ~{n_windows_done / max(n_supertiles_done,1):.0f}  "
                        f"rate={rate:.2f} sup/s  eta={eta_s/60:.1f} min",
                        flush=True,
                    )
                    t_last = now
            if args.limit_supertiles and n_supertiles_done >= args.limit_supertiles:
                break
        if args.limit_supertiles and n_supertiles_done >= args.limit_supertiles:
            break

    elapsed_min = (time.time() - t0) / 60
    print(f"Inference done: {n_supertiles_done} super-tiles, {n_windows_done} windows in {elapsed_min:.1f} min")

    # ----- Build pyramid -----
    pyramid_paths = ["0"]
    pyramid_scales = [(z_mm, y_mm, x_mm)]
    for ilev in range(1, args.build_pyramid + 1):
        factor = 2 ** ilev
        print(f"Building level {ilev} (yx /{factor})...")
        t_l = time.time()
        downsampled = downsample_uint8_yx(out0, factor)
        outL = root.create_array(
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
        print(f"  level {ilev}: shape={downsampled.shape}  built in {time.time() - t_l:.1f}s")

    # ----- OME-NGFF v0.5 metadata + provenance -----
    write_ome_ngff_v05(
        out_zarr_dir, pyramid_paths, pyramid_scales,
        xyz_orientation=input_orientation,
    )
    root_zj = out_zarr_dir / "zarr.json"
    rm = json.loads(root_zj.read_text())
    rm["attributes"]["_lumivox"] = {
        "input_zarr": str(args.input_zarr),
        "input_channel": args.channel,
        "input_level": args.level,
        "checkpoint": str(args.checkpoint),
        "patch_size": PATCH,
        "stride": args.stride,
        "blending": "gaussian" if use_blend else "none",
        "gaussian_sigma": args.gaussian_sigma if use_blend else None,
        "supertile_inner": inner,
        "halo": halo,
        "value_encoding": "uint8 probability * 255 (sigmoid output)",
        "model_arch": "ResidualEncoderUNet (frozen-encoder FT)",
        "z_score_normalization": "per-window, computed only over non-padded voxels",
    }
    root_zj.write_text(json.dumps(rm, indent=2))

    # ----- Zip to .ozx -----
    print(f"Packing to {args.output_ozx} ...")
    out_abs = args.output_ozx.absolute()
    if out_abs.exists():
        out_abs.unlink()
    src_root = out_zarr_dir
    with zipfile.ZipFile(out_abs, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(src_root.rglob("*")):
            arcname = path.relative_to(src_root).as_posix()
            if path.is_dir():
                zf.write(path, arcname=arcname + "/")
            else:
                zf.write(path, arcname=arcname)
    size_gb = out_abs.stat().st_size / 1e9
    print(f"Done: {out_abs}  ({size_gb:.2f} GB)")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
