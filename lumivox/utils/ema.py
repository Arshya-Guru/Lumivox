"""EMA update helpers for BYOL target network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


@torch.no_grad()
def update_target_ema(
    online: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    """Exponential moving average update: target <- tau*target + (1-tau)*online."""
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def cosine_ema_schedule(step: int, base_ema: float, max_steps: int) -> float:
    """Cosine EMA schedule: tau = 1 - (1-tau_base)*(cos(pi*k/K)+1)/2.

    At step 0: tau = base_ema.  At step max_steps: tau -> 1.0.
    """
    progress = min(step / max(max_steps, 1), 1.0)
    return 1.0 - (1.0 - base_ema) * 0.5 * (1.0 + math.cos(math.pi * progress))
