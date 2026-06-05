"""Topological loss for binary segmentation (``topo_ce``).

Dice + CE base loss plus a persistent-homology (PH) topological term, where the
**topological constraint is derived per-sample from the ground truth**: the GT
mask's Betti numbers (b0=components, b1=loops, b2=voids) are the target topology,
and the predicted foreground probability is pushed to match it.

    total = base_weight * dice_ce(logits, target)
          + topo_weight * topo(sigmoid(logits), target)

This is a focused binary port of the ``topo_ce`` loss from SynthTopo
(/nfs/khan/trainees/apooladi/topo/SynthTopo, synthtopo/losses.py::TopologicalLoss),
adapted from its multi-class form to our single-foreground Abeta case.

PH is computed with CubicalRipser (``cripser`` / ``tcripser``). If those aren't
installed, the topo term is disabled and this falls back to plain Dice+CE with a
one-time warning — so the rest of the pipeline still runs. Install with:

    pip install CubicalRipser        # provides both `cripser` and `tcripser`

The differentiable-barcode trick: the PH *structure* (which voxels are the
birth/death of each feature) is computed on a detached numpy copy, but the
persistence *values* are re-read from the live probability tensor at those
coordinates, so gradients flow back into the network.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from lumivox.training.finetune import dice_ce_loss


class TopoCELoss(nn.Module):
    """Binary Dice+CE + GT-supervised persistent-homology topological loss.

    Parameters
    ----------
    base_weight : float
        Weight on the Dice+CE base loss.
    topo_weight : float
        Weight on the PH topological term (SynthTopo's ``topo_ce`` default 1e-4).
    construction : str
        ``'0'`` = 6-connectivity (cripser), ``'N'`` = 26-connectivity (tcripser).
    thresh : float, optional
        If set, PH is computed only inside the foreground bounding box
        (foreground prob >= thresh), which bounds PH cost on large sparse volumes.
    enable_topo : bool
        Master switch; auto-disabled if cripser/tcripser are missing.
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        topo_weight: float = 1e-4,
        construction: str = "0",
        thresh: Optional[float] = None,
        enable_topo: bool = True,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.topo_weight = topo_weight
        self.construction = construction
        self.thresh = thresh
        self.enable_topo = enable_topo
        self._warned = False

        # Telemetry for logging (set each forward).
        self.last_base = torch.tensor(0.0)
        self.last_topo = torch.tensor(0.0)
        self.last_betti_err = torch.tensor(0.0)

        self.topo_available = self._check_deps()
        if self.enable_topo and not self.topo_available:
            warnings.warn(
                "TopoCELoss: cripser/tcripser not found — topo term disabled, "
                "falling back to Dice+CE. Install with `pip install CubicalRipser`."
            )
            self.enable_topo = False

    @staticmethod
    def _check_deps() -> bool:
        try:
            import cripser  # noqa: F401
            import tcripser  # noqa: F401
            return True
        except ImportError:
            return False

    # ── PH helpers ──────────────────────────────────────────────────────────

    def _compute_ph(self, vol_np: np.ndarray, maxdim: int):
        """Cubical PH barcode of a 3D float array (lower-star on the values)."""
        if self.construction == "N":
            import tcripser as trip
            return trip.computePH(vol_np, maxdim=maxdim)
        import cripser as crip
        return crip.computePH(vol_np, maxdim=maxdim)

    @staticmethod
    def _barcode_to_betti(bar, ndim: int) -> List[int]:
        """Betti numbers = count of essential (infinite-death) bars per dimension."""
        if bar is None or len(bar) == 0:
            return [0] * ndim
        inf = bar[bar[:, 2] == np.finfo(bar.dtype).max]
        return [int((inf[:, 0] == d).sum()) for d in range(ndim)]

    @staticmethod
    def _differentiable_persistence(vol: torch.Tensor, bar, ndim: int) -> List[torch.Tensor]:
        """Finite-bar persistence per dimension, sorted descending, differentiable.

        ``vol`` is the live tensor; birth/death values are re-read from it at the
        integer coordinates the (detached) barcode reports, so gradients flow.
        """
        empty = [vol.new_zeros(0) for _ in range(ndim)]
        if bar is None or len(bar) == 0:
            return empty
        fin = bar[bar[:, 2] < np.finfo(bar.dtype).max]
        if len(fin) == 0:
            return empty
        births = vol[tuple(fin[:, 3:3 + ndim].astype(np.int64).T)]
        deaths = vol[tuple(fin[:, 6:6 + ndim].astype(np.int64).T)]
        delta = deaths - births  # persistence (>= 0)
        out = []
        for d in range(ndim):
            dp = delta[fin[:, 0] == d]
            out.append(torch.sort(dp, descending=True)[0] if dp.numel() > 0 else vol.new_zeros(0))
        return out

    def _topo_loss(self, prob_fg: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """prob_fg, target: [B, D, H, W]. Returns scalar topo loss (batch mean)."""
        device = prob_fg.device
        B = prob_fg.shape[0]
        ndim = prob_fg.ndim - 1
        total = prob_fg.new_zeros(B)
        betti_err = 0.0

        for b in range(B):
            fg = prob_fg[b]
            gt = (target[b] > 0)

            # Optional ROI: restrict PH to the foreground bounding box.
            if self.thresh is not None:
                pts = torch.nonzero(fg >= self.thresh)
                if len(pts) > 0:
                    lo = pts.min(0).values
                    hi = pts.max(0).values + 1
                    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(ndim))
                    fg = fg[sl]
                    gt = gt[sl]

            # PH is computed on the COMPLEMENT (1 - prob) so that confident
            # foreground is "born" early (low filtration value).
            combos = 1.0 - fg
            combos_np = combos.detach().cpu().numpy().astype(float)
            pred_bar = self._compute_ph(combos_np, ndim)

            # Target topology from the GT mask (per-sample, supervised).
            gt_np = (1.0 - gt.float()).detach().cpu().numpy().astype(float)
            target_betti = self._barcode_to_betti(self._compute_ph(gt_np, ndim), ndim)

            persist = self._differentiable_persistence(combos, pred_bar, ndim)
            pred_betti = self._barcode_to_betti(pred_bar, ndim)

            a_term = prob_fg.new_zeros(())  # keep matched bars (push persistence -> 1)
            z_term = prob_fg.new_zeros(())  # suppress extra bars (push persistence -> 0)
            for d in range(ndim):
                p = persist[d]
                keep = min(int(target_betti[d]), p.numel())
                if keep > 0:
                    a_term = a_term + (1.0 - p[:keep]).sum()
                if p.numel() > keep:
                    z_term = z_term + p[keep:].sum()
                betti_err += abs(pred_betti[d] - target_betti[d])
            total[b] = a_term + z_term

        self.last_betti_err = torch.tensor(betti_err / max(1, B * ndim), device=device)
        return total.mean()

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits, target: [B, 1, D, H, W]. target binary in {0,1}."""
        base = dice_ce_loss(logits, target)
        self.last_base = base.detach()

        if not (self.enable_topo and self.topo_available):
            self.last_topo = torch.zeros((), device=logits.device)
            return self.base_weight * base

        prob_fg = torch.sigmoid(logits)[:, 0]      # [B, D, H, W]
        topo = self._topo_loss(prob_fg, target[:, 0])
        self.last_topo = topo.detach()
        return self.base_weight * base + self.topo_weight * topo
