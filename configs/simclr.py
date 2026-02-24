"""SimCLR-specific configuration overrides.

Uses shared encoder, projector, optimizer, and augmentation base.
SimCLR-specific: NT-Xent loss, symmetric augmentation, no EMA, no predictor.
"""

from configs.shared import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_OPTIMIZER_CONFIG,
    DEFAULT_SCHEDULE_CONFIG,
    ENCODER_CONFIG,
    PROJECTION_CONFIG,
)

SIMCLR_CONFIG = dict(
    method="simclr",
    encoder=ENCODER_CONFIG,
    projection=PROJECTION_CONFIG,
    optimizer=DEFAULT_OPTIMIZER_CONFIG,
    schedule=DEFAULT_SCHEDULE_CONFIG,
    data=DEFAULT_DATA_CONFIG,
    # SimCLR-specific
    loss=dict(
        name="ntxent",
        temperature=0.5,
        similarity_function="cosine",
    ),
    augmentation="symmetric",  # same blur prob for both views
    # No EMA, no predictor
    ema=None,
    predictor=None,
)

# Legacy optimizer for ablation
SIMCLR_LEGACY_OPTIMIZER = dict(
    name="sgd",
    lr=1e-2,
    weight_decay=3e-5,
    momentum=0.99,
    nesterov=True,
)


def get_config(
    epochs: int = 300,
    batch_size: int = 16,
    crop_size: int = 96,
    data_dir: str | None = None,
) -> dict:
    """Get full SimCLR training config."""
    cfg = dict(**SIMCLR_CONFIG)
    cfg["num_epochs"] = epochs
    cfg["data"] = dict(**cfg["data"], data_dir=data_dir)
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["crop_size"] = crop_size
    return cfg
