"""BYOL regression loss: 2 - 2 * cosine_similarity.

Used by nnBYOL3D and byol3d-legacy models.
Symmetrized: L = loss(q1, sg(z'2)) + loss(q2, sg(z'1))
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss.

    Args:
        x: online prediction q, shape [B, D].
        y: target projection z' (should be detached), shape [B, D].

    Returns:
        Scalar mean loss.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (2 - 2 * (x * y).sum(dim=-1)).mean()
