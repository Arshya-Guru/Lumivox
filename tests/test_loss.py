"""Loss computation tests."""

import pytest
import torch
import torch.nn.functional as F

from lumivox.losses.ntxent import NTXentLoss
from lumivox.losses.regression import regression_loss


class TestNTXentLoss:
    """Test NT-Xent contrastive loss."""

    def test_loss_decreases_for_similar_pairs(self):
        """Loss should be lower when positive pairs are more similar."""
        batch_size = 8
        dim = 32
        loss_fn = NTXentLoss(batch_size=batch_size, temperature=0.5, similarity_function="cosine")

        # Random (dissimilar) pairs
        z_i_random = F.normalize(torch.randn(batch_size, dim), dim=-1)
        z_j_random = F.normalize(torch.randn(batch_size, dim), dim=-1)
        loss_random, _ = loss_fn(z_i_random, z_j_random)

        # Similar pairs (z_j is close to z_i)
        z_i = F.normalize(torch.randn(batch_size, dim), dim=-1)
        z_j = z_i + torch.randn(batch_size, dim) * 0.01
        z_j = F.normalize(z_j, dim=-1)
        loss_similar, _ = loss_fn(z_i, z_j)

        assert loss_similar < loss_random

    def test_perfect_accuracy_for_identical(self):
        """When positive pairs are identical, accuracy should be high."""
        batch_size = 4
        dim = 16
        loss_fn = NTXentLoss(batch_size=batch_size, temperature=0.5, similarity_function="cosine")

        z = F.normalize(torch.randn(batch_size, dim), dim=-1)
        _, acc = loss_fn(z, z)
        assert acc > 0.5

    def test_output_is_scalar(self):
        batch_size = 4
        dim = 16
        loss_fn = NTXentLoss(batch_size=batch_size, temperature=0.5, similarity_function="cosine")

        z_i = F.normalize(torch.randn(batch_size, dim), dim=-1)
        z_j = F.normalize(torch.randn(batch_size, dim), dim=-1)
        loss, acc = loss_fn(z_i, z_j)
        assert loss.dim() == 0
        assert isinstance(acc, float)


class TestRegressionLoss:
    """Test BYOL regression loss."""

    def test_zero_for_identical(self):
        """Loss should be 0 when inputs are identical (after normalization)."""
        x = torch.randn(8, 32)
        loss = regression_loss(x, x)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        """Loss should be positive for different inputs."""
        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        loss = regression_loss(x, y)
        assert loss > 0

    def test_max_for_opposite(self):
        """Loss should be maximal (close to 4) for opposite vectors."""
        x = torch.randn(8, 32)
        y = -x
        loss = regression_loss(x, y)
        assert torch.allclose(loss, torch.tensor(4.0), atol=0.1)

    def test_output_is_scalar(self):
        x = torch.randn(4, 16)
        y = torch.randn(4, 16)
        loss = regression_loss(x, y)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        x = torch.randn(4, 16, requires_grad=True)
        y = torch.randn(4, 16)
        loss = regression_loss(x, y)
        loss.backward()
        assert x.grad is not None
