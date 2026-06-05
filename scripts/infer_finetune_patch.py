"""Run FT segmentation inference on a single patch and save a QC triptych.

Loads a Lightning FT checkpoint, picks one patch from the FT manifest, runs
forward, computes dice, and saves a PNG showing image | GT | pred as max
intensity projections along the three axes.

Usage:
    python scripts/infer_finetune_patch.py \
        --checkpoint checkpoints/finetune_abeta_nnbyol3d_frozen/finetune-ep0064-dice0.7350.ckpt \
        --manifest manifests/abeta_ft_first_pass.json \
        --patch-id sub-AS39Me1_patch00 \
        --output qc/infer_nnbyol3d_AS39Me1_patch00.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `from lumivox.X import Y` work when this script is run by path,
# not as a module — same trick the rest of the scripts/ dir effectively gets
# through `python -m lumivox.training.X`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lumivox.training.finetune import build_segmentation_model


def load_lightning_state_dict(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Lightning wraps the model as self.model — strip "model." prefix.
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / sd if sd >= eps else x - mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--patch-id", required=True)
    p.add_argument("--output", required=True, help="PNG output path")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entry = next((e for e in manifest["patches"] if e["patch_id"] == args.patch_id), None)
    if entry is None:
        raise SystemExit(f"Patch {args.patch_id} not in manifest")

    print(f"Patch:    {args.patch_id}  (subject: {entry['subject_id']}, dataset: {entry['dataset']})")
    print(f"Image:    {entry['raw_path']}")
    print(f"GT mask:  {entry['seg_gold_path']}")

    img_nii = nib.load(entry["raw_path"])
    img = img_nii.get_fdata().astype(np.float32)
    msk = nib.load(entry["seg_gold_path"]).get_fdata().astype(np.float32)
    while img.ndim > 3 and img.shape[0] == 1: img = img[0]
    while msk.ndim > 3 and msk.shape[0] == 1: msk = msk[0]
    print(f"Image shape: {img.shape}  mask shape: {msk.shape}")
    print(f"Image intensity range: [{img.min():.1f}, {img.max():.1f}]  mean={img.mean():.1f}")
    print(f"Mask foreground voxels: {int((msk > 0).sum())} / {msk.size} ({100*(msk>0).mean():.3f}%)")

    img_z = zscore(img)
    msk_bin = (msk > 0).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building model on {device}")
    model = build_segmentation_model(checkpoint_path=None, num_classes=1, deep_supervision=True, dropout=0.0)
    sd = load_lightning_state_dict(args.checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"State dict: {len(sd)} keys loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing examples: {missing[:3]}")
    if unexpected:
        print(f"  unexpected examples: {unexpected[:3]}")

    model.eval().to(device)
    x = torch.from_numpy(img_z).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    print(f"Forward pass...")
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
    probs = torch.sigmoid(out)[0, 0].cpu().numpy()
    pred = (probs > args.threshold).astype(np.float32)

    # Dice
    inter = float((pred * msk_bin).sum())
    union = float(pred.sum() + msk_bin.sum())
    dice = (2 * inter + 1) / (union + 1)
    print(f"Predicted foreground voxels: {int(pred.sum())} ({100*pred.mean():.3f}%)")
    print(f"DICE on this patch: {dice:.4f}")

    # Visualization: 3 rows (axis 0,1,2 MIPs) × 3 cols (image / GT / pred over image)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axis_names = ["Z (D)", "Y (H)", "X (W)"]
    for row, ax_idx in enumerate(range(3)):
        img_mip = img.max(axis=ax_idx)
        gt_mip = msk_bin.max(axis=ax_idx)
        pred_mip = pred.max(axis=ax_idx)

        # Col 0: image only
        axes[row, 0].imshow(img_mip, cmap="gray")
        axes[row, 0].set_title(f"image MIP along {axis_names[row]}")

        # Col 1: image + GT overlay
        axes[row, 1].imshow(img_mip, cmap="gray")
        axes[row, 1].imshow(np.ma.masked_where(gt_mip == 0, gt_mip), cmap="autumn", alpha=0.6)
        axes[row, 1].set_title(f"GT overlay")

        # Col 2: image + pred overlay
        axes[row, 2].imshow(img_mip, cmap="gray")
        axes[row, 2].imshow(np.ma.masked_where(pred_mip == 0, pred_mip), cmap="winter", alpha=0.6)
        axes[row, 2].set_title(f"PRED overlay (thr={args.threshold})")

        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    fig.suptitle(
        f"{args.patch_id}  |  ckpt: {Path(args.checkpoint).name}  |  DICE={dice:.4f}",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved PNG:  {out_path}")

    # Also write probability map (float32) and binary mask (uint8) as NIfTI
    # in the same affine as the source image so they overlay cleanly in viewers.
    affine = img_nii.affine
    header = img_nii.header.copy()
    prob_path = out_path.with_name(out_path.stem + "_prob.nii.gz")
    mask_path = out_path.with_name(out_path.stem + "_mask.nii.gz")
    nib.save(nib.Nifti1Image(probs.astype(np.float32), affine, header), str(prob_path))
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine, header), str(mask_path))
    print(f"Saved prob: {prob_path}")
    print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()
