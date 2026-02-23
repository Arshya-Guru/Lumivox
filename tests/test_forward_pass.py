"""Shape tests and gradient flow checks for all 3 models.

Uses tiny configs (small channels, small spatial) for fast testing.
"""

import pytest
import torch
import torch.nn as nn

from lumivox.models.simclr_model import SimCLRModel
from lumivox.models.nnbyol3d_model import NnBYOL3DModel
from lumivox.models.byol3d_legacy_model import BYOL3DLegacyModel
from lumivox.utils.ema import update_target_ema


class TestSimCLRForward:
    """Test SimCLR model forward pass."""

    def test_output_shape(self):
        model = SimCLRModel(num_input_channels=1, proj_hidden_dim=64, proj_output_dim=32)
        x = torch.randn(2, 1, 32, 32, 32)
        z = model(x)
        assert z.shape == (2, 32)

    def test_output_normalized(self):
        model = SimCLRModel(num_input_channels=1, proj_hidden_dim=64, proj_output_dim=32)
        x = torch.randn(2, 1, 32, 32, 32)
        z = model(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradients_flow(self):
        model = SimCLRModel(num_input_channels=1, proj_hidden_dim=64, proj_output_dim=32)
        x = torch.randn(2, 1, 32, 32, 32)
        z = model(x)
        loss = z.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert not torch.all(p.grad == 0), f"Zero grad for {name}"

    def test_encode_method(self):
        model = SimCLRModel(num_input_channels=1, proj_hidden_dim=64, proj_output_dim=32)
        x = torch.randn(2, 1, 32, 32, 32)
        h = model.encode(x)
        assert h.shape == (2, 320)


class TestNnBYOL3DForward:
    """Test nnBYOL3D model forward pass."""

    def test_output_shapes(self):
        model = NnBYOL3DModel.create_pair(
            num_input_channels=1,
            proj_hidden_dim=64,
            proj_output_dim=32,
            pred_hidden_dim=64,
        )
        v1 = torch.randn(2, 1, 32, 32, 32)
        v2 = torch.randn(2, 1, 32, 32, 32)
        o1, o2, t1, t2 = model(v1, v2)

        assert o1["projection"].shape == (2, 32)
        assert o1["prediction"].shape == (2, 32)
        assert o2["projection"].shape == (2, 32)
        assert o2["prediction"].shape == (2, 32)
        assert t1["projection"].shape == (2, 32)
        assert t2["projection"].shape == (2, 32)

    def test_target_frozen(self):
        model = NnBYOL3DModel.create_pair(
            num_input_channels=1,
            proj_hidden_dim=64,
            proj_output_dim=32,
            pred_hidden_dim=64,
        )
        for p in model.target.parameters():
            assert not p.requires_grad

    def test_gradients_flow_online_only(self):
        model = NnBYOL3DModel.create_pair(
            num_input_channels=1,
            proj_hidden_dim=64,
            proj_output_dim=32,
            pred_hidden_dim=64,
        )
        v1 = torch.randn(2, 1, 32, 32, 32)
        v2 = torch.randn(2, 1, 32, 32, 32)
        o1, o2, t1, t2 = model(v1, v2)

        loss = o1["prediction"].sum() + o2["prediction"].sum()
        loss.backward()

        # Online should have gradients
        for name, p in model.online.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for online.{name}"

    def test_ema_updates_target(self):
        model = NnBYOL3DModel.create_pair(
            num_input_channels=1,
            proj_hidden_dim=64,
            proj_output_dim=32,
            pred_hidden_dim=64,
        )
        # Store target params before
        target_before = {
            k: v.clone() for k, v in model.target.state_dict().items()
        }

        # Modify online params to be different
        for p in model.online.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)

        # EMA update
        update_target_ema(model.online, model.target, tau=0.99)

        # Target should have changed
        changed = False
        for k, v in model.target.state_dict().items():
            if not torch.allclose(v, target_before[k]):
                changed = True
                break
        assert changed, "EMA did not change target params"


class TestBYOL3DLegacyForward:
    """Test legacy BYOL3D model forward pass."""

    def test_output_shapes(self):
        model = BYOL3DLegacyModel.create_pair(
            in_channels=1,
            base_channels=8,
            num_levels=3,
            projector_hidden_dim=32,
            projector_output_dim=16,
            predictor_hidden_dim=32,
        )
        v1 = torch.randn(2, 1, 16, 16, 16)
        v2 = torch.randn(2, 1, 16, 16, 16)
        o1, o2, t1, t2 = model(v1, v2)

        assert o1["projection"].shape == (2, 16)
        assert o1["prediction"].shape == (2, 16)
        assert t1["projection"].shape == (2, 16)

    def test_gradients_flow(self):
        model = BYOL3DLegacyModel.create_pair(
            in_channels=1,
            base_channels=8,
            num_levels=3,
            projector_hidden_dim=32,
            projector_output_dim=16,
            predictor_hidden_dim=32,
        )
        v1 = torch.randn(2, 1, 16, 16, 16)
        v2 = torch.randn(2, 1, 16, 16, 16)
        o1, o2, t1, t2 = model(v1, v2)

        loss = o1["prediction"].sum() + o2["prediction"].sum()
        loss.backward()

        for name, p in model.online.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for online.{name}"
