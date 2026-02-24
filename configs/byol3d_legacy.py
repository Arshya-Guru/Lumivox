"""Legacy BYOL3D configuration (preserved from original repo).

Uses the old UNetEncoder3D, old augmentations, old fine-tuner.
NOT directly comparable to simclr/nnbyol3d due to different encoder
and fine-tuning pipeline.
"""

BYOL3D_LEGACY_CONFIG = dict(
    method="byol3d-legacy",
    encoder=dict(
        name="UNetEncoder3D",
        in_channels=1,
        base_channels=32,
        num_levels=5,
        channels=[32, 64, 128, 256, 320],
        use_residuals=True,
    ),
    projection=dict(
        input_dim=320,
        hidden_dim=2048,  # original used 2048, not 4096
        output_dim=256,
    ),
    predictor=dict(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
    ),
    optimizer=dict(
        name="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    ),
    schedule=dict(
        name="cosine",
        warmup_epochs=10,
    ),
    data=dict(
        crop_size=96,
        min_overlap_fraction=0.4,  # legacy used 0.4 volumetric overlap
        batch_size=16,
        num_workers=4,
    ),
    loss=dict(name="regression"),
    augmentation="legacy_asymmetric",
    ema=dict(
        base_ema=0.99,
        schedule="cosine",
    ),
)


def get_config(
    epochs: int = 300,
    batch_size: int = 16,
    crop_size: int = 96,
    data_dir: str | None = None,
) -> dict:
    """Get full legacy BYOL3D training config."""
    cfg = dict(**BYOL3D_LEGACY_CONFIG)
    cfg["num_epochs"] = epochs
    cfg["data"] = dict(**cfg["data"], data_dir=data_dir)
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["crop_size"] = crop_size
    return cfg
