"""NT-Xent contrastive loss for SimCLR.

Ported from nnssl's contrastive_loss.py with minimal changes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss.

    Args:
        batch_size: number of samples per view (total pairs = 2 * batch_size).
        temperature: softmax temperature (default 0.5).
        similarity_function: 'cosine' or 'dot'.
        device: torch device for mask tensors.
    """

    def __init__(
        self,
        batch_size: int,
        temperature: float = 0.5,
        similarity_function: str = "cosine",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cpu")

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self._cosine_similarity = nn.CosineSimilarity(dim=-1)

        if similarity_function == "cosine":
            self.similarity_function = self._cosine_sim
        elif similarity_function == "dot":
            self.similarity_function = self._dot_sim
        else:
            raise ValueError(
                f"Invalid similarity function: {similarity_function}. "
                "Supported: 'cosine', 'dot'."
            )

        self.mask_samples_from_same_repr = (
            self._get_correlated_mask().bool().to(self.device)
        )

    def _get_correlated_mask(self) -> torch.Tensor:
        N = 2 * self.batch_size
        diag = np.eye(N)
        l1 = np.eye(N, N, k=-self.batch_size)
        l2 = np.eye(N, N, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).bool()
        return mask

    def _cosine_sim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

    @staticmethod
    def _dot_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Compute NT-Xent loss.

        Args:
            z_i: L2-normalized projections from view 1, shape [B, D].
            z_j: L2-normalized projections from view 2, shape [B, D].

        Returns:
            (loss, accuracy) tuple.
        """
        z_i = z_i.to(self.device)
        z_j = z_j.to(self.device)

        representations = torch.cat([z_j, z_i], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size, dtype=torch.long, device=self.device)

        loss = self.criterion(logits, labels)
        accuracy = (
            (torch.max(logits.detach(), dim=1)[1] == 0).sum().item() / logits.size(0)
        )

        return loss / (2 * self.batch_size), accuracy
