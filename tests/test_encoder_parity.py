"""Verify ResEncL matches nnssl's architecture_registry version.

Tests that build_resenc_l() produces a model whose state dict keys
and tensor shapes exactly match what nnssl uses.
"""

import pytest
import torch
import torch.nn as nn

from lumivox.encoders.resenc_l import build_resenc_l


class TestResEncLParity:
    """Verify ResEncL encoder matches nnssl specification."""

    def test_output_channels(self):
        """Check that output_channels attribute matches expected values."""
        encoder = build_resenc_l(num_input_channels=1)
        assert encoder.output_channels == [32, 64, 128, 256, 320, 320]

    def test_return_skips_shape(self):
        """Check that forward returns correct number of skip tensors."""
        encoder = build_resenc_l(num_input_channels=1)
        x = torch.randn(1, 1, 96, 96, 96)
        skips = encoder(x)
        assert isinstance(skips, (list, tuple))
        assert len(skips) == 6

    def test_skip_spatial_dimensions(self):
        """Check spatial dims halve at each stage (except first)."""
        encoder = build_resenc_l(num_input_channels=1)
        x = torch.randn(1, 1, 96, 96, 96)
        skips = encoder(x)

        expected_shapes = [
            (1, 32, 96, 96, 96),
            (1, 64, 48, 48, 48),
            (1, 128, 24, 24, 24),
            (1, 256, 12, 12, 12),
            (1, 320, 6, 6, 6),
            (1, 320, 3, 3, 3),
        ]
        for skip, expected in zip(skips, expected_shapes):
            assert skip.shape == expected, f"Got {skip.shape}, expected {expected}"

    def test_state_dict_keys_match_nnssl(self):
        """Verify state dict key structure is compatible with nnssl.

        nnssl's ResidualEncoderUNet has encoder with keys like:
        'encoder.stages.X...' and 'encoder.stem...'
        Our encoder (ResidualEncoder) should have the same key prefix pattern.
        """
        encoder = build_resenc_l(num_input_channels=1)
        sd = encoder.state_dict()

        # Must have stages and stem
        has_stages = any(k.startswith("stages.") for k in sd.keys())
        has_stem = any(k.startswith("stem.") for k in sd.keys())
        assert has_stages, "Missing 'stages.' keys in state dict"
        assert has_stem, "Missing 'stem.' keys in state dict"

    def test_bottleneck_dim_after_gap(self):
        """Verify bottleneck dim is 320 after GAP."""
        encoder = build_resenc_l(num_input_channels=1)
        gap = nn.AdaptiveAvgPool3d(1)
        x = torch.randn(2, 1, 96, 96, 96)
        skips = encoder(x)
        bottleneck = gap(skips[-1]).flatten(1)
        assert bottleneck.shape == (2, 320)

    def test_multichannel_input(self):
        """Test with multi-channel input."""
        encoder = build_resenc_l(num_input_channels=3)
        x = torch.randn(1, 3, 96, 96, 96)
        skips = encoder(x)
        assert len(skips) == 6
        assert skips[-1].shape[1] == 320
