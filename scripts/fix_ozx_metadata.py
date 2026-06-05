"""Patch the OME-NGFF metadata of an existing .ozx file produced by
scripts/infer_finetune_brain.py (v1 or early-v2) so it loads correctly in
zarrnii / napari-ome-zarr.

Fixes:
  1. Scale values are divided by 1000 (we wrote them in micrometers; tools read
     them as millimeters). After the fix, scale values are in millimeters and
     the unit field is set to 'millimeter' to match.
  2. Adds an xyz_orientation field at the root attributes (defaults to "IAR",
     matching the input we used for the AS36F4 runs; override with --orientation).

It does NOT change the underlying array dimensionality (the existing files are
3D [z, y, x]; this patch keeps them 3D but corrects the units + orientation).
If the consuming tool requires 4D [c, z, y, x] arrays, the file must be
regenerated from inference — that is what the updated script does for future
runs.

Usage:
    python scripts/fix_ozx_metadata.py path/to/file.ozx [--orientation IAR] [--in-place]
        --in-place    Overwrite the original file. Otherwise writes
                      <basename>_fixed.ozx next to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def patch_root_meta(root_meta: dict, xyz_orientation: str | None) -> dict:
    """Return a new root zarr.json dict with corrected OME metadata."""
    attrs = root_meta.get("attributes", {})
    ome = attrs.get("ome", {})
    multiscales = ome.get("multiscales", [])
    for ms in multiscales:
        for ax in ms.get("axes", []):
            if ax.get("type") == "space":
                ax["unit"] = "millimeter"
        for ds in ms.get("datasets", []):
            for tx in ds.get("coordinateTransformations", []):
                if tx.get("type") == "scale":
                    tx["scale"] = [float(v) / 1000.0 for v in tx["scale"]]
    if xyz_orientation:
        attrs["xyz_orientation"] = xyz_orientation
    attrs["_lumivox_patch"] = {
        "tool": "scripts/fix_ozx_metadata.py",
        "actions": [
            "divided all spatial scale values by 1000 (um -> mm)",
            "set axis unit to 'millimeter'",
            f"set xyz_orientation = '{xyz_orientation}'" if xyz_orientation else "no orientation set",
        ],
    }
    root_meta["attributes"] = attrs
    return root_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Input .ozx file to patch")
    ap.add_argument("--orientation", default="IAR",
                    help="xyz_orientation to declare. Default IAR (matches the AS36F4 input).")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input file. Otherwise writes <basename>_fixed.ozx.")
    args = ap.parse_args()

    src = args.input
    if not src.exists() or not src.is_file():
        raise SystemExit(f"Input not found or not a file: {src}")

    if args.in_place:
        dst = src
    else:
        dst = src.with_name(src.stem + "_fixed.ozx")

    print(f"Patching: {src}")
    print(f"  output: {dst}")

    # Build the patched zip in a temp file, then move to destination.
    with tempfile.NamedTemporaryFile(prefix="ozxpatch_", suffix=".ozx", delete=False) as tf:
        tmp_out = Path(tf.name)

    try:
        with zipfile.ZipFile(src, "r") as zin, \
             zipfile.ZipFile(tmp_out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
            patched = False
            for info in zin.infolist():
                with zin.open(info) as fin:
                    data = fin.read()
                if info.filename == "zarr.json":
                    root_meta = json.loads(data)
                    root_meta = patch_root_meta(root_meta, xyz_orientation=args.orientation)
                    data = json.dumps(root_meta, indent=2).encode("utf-8")
                    new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                    new_info.compress_type = zipfile.ZIP_STORED
                    zout.writestr(new_info, data)
                    patched = True
                    print(f"  patched root zarr.json ({len(data)} bytes)")
                else:
                    # Preserve original mode (file vs dir).
                    new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                    new_info.compress_type = zipfile.ZIP_STORED
                    new_info.external_attr = info.external_attr
                    zout.writestr(new_info, data)
            if not patched:
                raise SystemExit("No root zarr.json found in archive — is this really an ozx?")

        shutil.move(str(tmp_out), str(dst))
        size = dst.stat().st_size
        print(f"Done. Wrote {size/1e9:.2f} GB")
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
