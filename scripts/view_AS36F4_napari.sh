#!/bin/bash
# Launch napari with the AS36F4 4 um isotropic image (channel 0 = Abeta) and
# the binarized lumivox FT nnBYOL3D mask overlaid.
#
# Both inputs are OME-Zarr v3 (one zipped as .ozx for spimquant's outputs, but
# here we use the unzipped .ome.zarr mask we just built). napari opens them via
# the napari-ome-zarr plugin.
#
# Usage:  bash scripts/view_AS36F4_napari.sh
#
# Notes:
#   - The input is a 3-channel volume; napari will split it into one layer per
#     channel. Channel 0 is Abeta — keep that visible and hide channels 1 and 2.
#   - The mask is binary uint8 (1 = foreground). It opens as an Image layer
#     by default; in the GUI right-click -> "Convert to Labels" if you want a
#     proper segmentation overlay (or just set its colormap to something
#     contrasting like "red" and blend mode to "additive").

set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

INPUT_OME_ZARR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_res-4um_SPIM.ome.zarr
MASK_OME_ZARR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350Thr0p5_mask.ome.zarr

echo "Image: $INPUT_OME_ZARR"
echo "Mask:  $MASK_OME_ZARR"

# Sanity check inputs exist
for p in "$INPUT_OME_ZARR" "$MASK_OME_ZARR"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing path: $p" >&2
        exit 1
    fi
done

pixi run napari "$INPUT_OME_ZARR" "$MASK_OME_ZARR"
