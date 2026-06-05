"""Ensemble inference for the v4 fine-tuning sweep.

Basic call:
    python scripts/inference.py <input.nii.gz> --model sim_frz20 [--overlap]

--model selects a 5-seed ensemble: {sim|nnb}_{frz|unf}{10|20}
    sim/nnb -> simclr / nnbyol3d encoder
    frz/unf -> frozen / unfrozen
    10/20   -> 0.1 / 0.2 decoder dropout
e.g. sim_frz20 = simclr, frozen encoder, 0.2 dropout -> the 5 runs
     checkpoints/ft_v4_sweep/ftv4_simclr_d20_frozen_s{0..4}/ (best-val/dice ckpt each).

Each of the 5 members predicts a BINARY mask (sigmoid > 0.5); the ensemble
probability is the fraction of members voting foreground (= 0.2*sum when all 5
are present). Outputs land next to the input:
    <stem>_prob.nii.gz       float32, values in {0, 0.2, ..., 1.0}
    <stem>_maskthr05.nii.gz  uint8, prob >= 0.5  (majority vote)

Patch handling:
    128^3 input          -> single forward per member (overlap ignored).
    larger input         -> 128^3 sliding window:
        default          -> non-overlapping tiles (stride 128)
        --overlap        -> stride 64 (50% overlap), Gaussian-blended
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumivox.training.finetune import build_segmentation_model

PATCH = 128
SWEEP_ROOT = Path("checkpoints/ft_v4_sweep")

ENC = {"sim": "simclr", "nnb": "nnbyol3d"}
FRZ = {"frz": "frozen", "unf": "unfrozen"}
DROP = {"10": "d10", "20": "d20"}


def parse_model(m: str):
    """'sim_frz20' -> ('simclr', 'frozen', 'd20')."""
    try:
        enc_key, rest = m.split("_")
        enc, frz, drop = ENC[enc_key], FRZ[rest[:3]], DROP[rest[3:]]
    except (ValueError, KeyError):
        raise SystemExit(
            f"bad --model '{m}'. Expected {{sim|nnb}}_{{frz|unf}}{{10|20}}, e.g. sim_frz20"
        )
    return enc, frz, drop


def best_ckpt(run_dir: Path):
    """Highest-val/dice checkpoint in a run dir (or None)."""
    cks = glob.glob(str(run_dir / "finetune-epoch*.ckpt"))
    if not cks:
        return None
    def dice_of(p):
        m = re.search(r"dice([0-9.]+)\.ckpt", p)
        return float(m.group(1)) if m else -1.0
    return max(cks, key=dice_of)


def load_member(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    model = build_segmentation_model(
        checkpoint_path=None,
        num_classes=hp.get("num_classes", 1),
        deep_supervision=hp.get("deep_supervision", True),
        dropout=hp.get("dropout", 0.0),
    )
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / sd if sd >= eps else x - mu


def gaussian_kernel(p: int, sigma: float = None) -> np.ndarray:
    sigma = sigma or p / 8.0
    c = np.arange(p, dtype=np.float32) - (p - 1) / 2.0
    g = np.exp(-(c ** 2) / (2 * sigma ** 2))
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    k /= k.max()
    return np.maximum(k, 1e-3).astype(np.float32)


def _positions(n: int, stride: int) -> list:
    if n <= PATCH:
        return [0]
    ps = list(range(0, n - PATCH + 1, stride))
    if ps[-1] != n - PATCH:
        ps.append(n - PATCH)
    return ps


@torch.no_grad()
def predict_prob(model, vol: np.ndarray, device, stride: int, gauss: bool) -> np.ndarray:
    """Stitched foreground-probability map for one member over an arbitrary volume."""
    D, H, W = vol.shape
    if (D, H, W) == (PATCH, PATCH, PATCH):
        x = torch.from_numpy(zscore(vol)).float()[None, None].to(device)
        out = model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return torch.sigmoid(out)[0, 0].float().cpu().numpy()

    prob_acc = np.zeros((D, H, W), np.float32)
    w_acc = np.zeros((D, H, W), np.float32)
    weight = gaussian_kernel(PATCH) if gauss else np.ones((PATCH, PATCH, PATCH), np.float32)
    for z in _positions(D, stride):
        for y in _positions(H, stride):
            for x in _positions(W, stride):
                z2, y2, x2 = min(z + PATCH, D), min(y + PATCH, H), min(x + PATCH, W)
                sub = vol[z:z2, y:y2, x:x2]
                win = np.zeros((PATCH, PATCH, PATCH), np.float32)
                win[: z2 - z, : y2 - y, : x2 - x] = zscore(sub)  # per-window norm (matches training)
                xt = torch.from_numpy(win)[None, None].to(device)
                out = model(xt)
                out = out[0] if isinstance(out, (list, tuple)) else out
                p = torch.sigmoid(out)[0, 0].float().cpu().numpy()
                prob_acc[z:z2, y:y2, x:x2] += (p * weight)[: z2 - z, : y2 - y, : x2 - x]
                w_acc[z:z2, y:y2, x:x2] += weight[: z2 - z, : y2 - y, : x2 - x]
    return np.where(w_acc > 1e-6, prob_acc / np.maximum(w_acc, 1e-6), 0.0)


def main():
    ap = argparse.ArgumentParser(description="v4 ensemble inference")
    ap.add_argument("input", type=str, help="input .nii.gz (any size; tiled if > 128^3)")
    ap.add_argument("--model", required=True, help="{sim|nnb}_{frz|unf}{10|20}, e.g. sim_frz20")
    ap.add_argument("--overlap", action="store_true",
                    help="50%% sliding-window overlap (stride 64, Gaussian-blended). "
                         "Default: non-overlapping tiles. Ignored for 128^3 inputs.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="per-member foreground threshold (default 0.5)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    enc, frz, drop = parse_model(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve the 5-member ensemble.
    members = []
    for s in range(5):
        run = SWEEP_ROOT / f"ftv4_{enc}_{drop}_{frz}_s{s}"
        ck = best_ckpt(run)
        if ck:
            members.append((s, ck))
        else:
            print(f"  WARNING: no checkpoint for s{s} ({run.name}) — skipping")
    if not members:
        raise SystemExit(f"No checkpoints found for model '{args.model}' under {SWEEP_ROOT}")
    print(f"Model {args.model} -> {enc}/{frz}/{drop}; using {len(members)}/5 members: "
          f"{[s for s, _ in members]}")

    inp = Path(args.input)
    img = nib.load(str(inp))
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = vol[0]
    D, H, W = vol.shape
    is_patch = (D, H, W) == (PATCH, PATCH, PATCH)
    stride = PATCH if (is_patch or not args.overlap) else PATCH // 2
    print(f"Input {inp.name}: shape={vol.shape}  "
          f"mode={'single-forward' if is_patch else f'sliding-window stride={stride}'}")

    # Ensemble: each member's binary vote, accumulated.
    vote = np.zeros((D, H, W), np.float32)
    for s, ck in members:
        prob = predict_prob(load_member(ck, device), vol, device,
                            stride=stride, gauss=(not is_patch and args.overlap))
        vote += (prob > args.threshold).astype(np.float32)
        print(f"  member s{s}: foreground voxels = {int((prob > args.threshold).sum())}")

    ens_prob = (vote / len(members)).astype(np.float32)      # {0, 1/N, ..., 1}
    ens_mask = (ens_prob >= 0.5).astype(np.uint8)            # majority vote

    base = inp.name[:-7] if inp.name.endswith(".nii.gz") else inp.stem
    out_dir = inp.parent / args.model           # per-model subdir, e.g. .../sub-AS40F2/sim_frz20/
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_path = out_dir / (base + "_prob.nii.gz")
    mask_path = out_dir / (base + "_maskthr05.nii.gz")
    aff, hdr = img.affine, img.header.copy()
    nib.save(nib.Nifti1Image(ens_prob, aff, hdr), str(prob_path))
    nib.save(nib.Nifti1Image(ens_mask, aff, hdr), str(mask_path))
    print(f"Wrote {prob_path}")
    print(f"Wrote {mask_path}  (foreground voxels: {int(ens_mask.sum())})")


if __name__ == "__main__":
    main()
