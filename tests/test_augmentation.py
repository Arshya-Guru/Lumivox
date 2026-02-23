"""Verify augmentation pipeline correctness."""

import pytest
import torch

from lumivox.augmentations.shared_aug import (
    SharedAugmentation,
    get_augmentation_config,
    simulate_low_resolution,
    brightness_gradient_3d,
    gaussian_blur_3d,
)


class TestSimulateLowResolution:
    """Verify SimulateLowResolution produces correct output shape."""

    def test_output_shape_preserved(self):
        x = torch.randn(2, 1, 32, 32, 32)
        out = simulate_low_resolution(x, zoom_range=(0.5, 1.0), apply_prob=1.0)
        assert out.shape == x.shape

    def test_no_op_at_zoom_1(self):
        x = torch.randn(2, 1, 32, 32, 32)
        out = simulate_low_resolution(x, zoom_range=(1.0, 1.0), apply_prob=1.0)
        # When zoom=1.0, output should be unchanged
        assert torch.allclose(out, x)


class TestBrightnessGradient:
    """Verify brightness gradient is applied along a random axis."""

    def test_output_shape_preserved(self):
        x = torch.randn(2, 1, 16, 16, 16)
        out = brightness_gradient_3d(x, strength_range=(0.1, 0.1), apply_prob=1.0)
        assert out.shape == x.shape

    def test_gradient_is_applied(self):
        torch.manual_seed(42)
        x = torch.zeros(1, 1, 16, 16, 16)
        out = brightness_gradient_3d(x, strength_range=(0.3, 0.3), apply_prob=1.0)
        # Should not be all zeros anymore
        assert not torch.allclose(out, x)


class TestGaussianBlur:
    """Verify Gaussian blur preserves shape."""

    def test_output_shape_preserved(self):
        x = torch.randn(2, 1, 16, 16, 16)
        out = gaussian_blur_3d(x, sigma_range=(0.5, 0.5), apply_prob=1.0)
        assert out.shape == x.shape


class TestAugmentationConfig:
    """Verify symmetric vs asymmetric blur configurations."""

    def test_simclr_symmetric(self):
        cfg = get_augmentation_config("simclr")
        v1_blur = cfg["view1"]["gaussian_blur_cfg"]["apply_prob"]
        v2_blur = cfg["view2"]["gaussian_blur_cfg"]["apply_prob"]
        assert v1_blur == v2_blur == 0.5

    def test_nnbyol3d_asymmetric(self):
        cfg = get_augmentation_config("nnbyol3d")
        v1_blur = cfg["view1"]["gaussian_blur_cfg"]["apply_prob"]
        v2_blur = cfg["view2"]["gaussian_blur_cfg"]["apply_prob"]
        assert v1_blur == 0.5
        assert v2_blur == 0.1

    def test_shared_augmentation_callable(self):
        cfg = get_augmentation_config("simclr")
        aug = SharedAugmentation(**cfg["view1"])
        x = torch.randn(1, 1, 16, 16, 16)
        out = aug(x)
        assert out.shape == x.shape

    def test_both_methods_have_all_augmentations(self):
        """Both methods should have the same set of augmentation keys."""
        simclr_cfg = get_augmentation_config("simclr")
        byol_cfg = get_augmentation_config("nnbyol3d")

        for view in ("view1", "view2"):
            simclr_keys = set(simclr_cfg[view].keys())
            byol_keys = set(byol_cfg[view].keys())
            assert simclr_keys == byol_keys, (
                f"Mismatch in {view}: {simclr_keys.symmetric_difference(byol_keys)}"
            )
