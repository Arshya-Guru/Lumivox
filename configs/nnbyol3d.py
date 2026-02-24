"""nnBYOL3D-specific configuration overrides.

Uses shared encoder, projector, optimizer, and augmentation base.
nnBYOL3D-specific: regression loss, EMA target, predictor MLP,
asymmetric blur probabilities.
"""

from configs.shared import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_OPTIMIZER_CONFIG,
    DEFAULT_SCHEDULE_CONFIG,
    ENCODER_CONFIG,
    PROJECTION_CONFIG,
)

NNBYOL3D_CONFIG = dict(
    method="nnbyol3d",
    encoder=ENCODER_CONFIG,
    projection=PROJECTION_CONFIG,
    optimizer=DEFAULT_OPTIMIZER_CONFIG,
    schedule=DEFAULT_SCHEDULE_CONFIG,
    data=DEFAULT_DATA_CONFIG,
    # nnBYOL3D-specific
    loss=dict(
        name="regression",
        # 2 - 2*cosine_similarity, symmetrized
    ),
    augmentation="asymmetric",  # view1 blur=0.5, view2 blur=0.1
    # EMA target network
    ema=dict(
        base_ema_300=0.99,  # for <=300 epochs
        base_ema_1000=0.996,  # for 1000 epochs
        schedule="cosine",
    ),
    # Predictor MLP (same architecture as projector)
    predictor=dict(
        input_dim=256,
        hidden_dim=4096,
        output_dim=256,
    ),
)


def get_config(
    epochs: int = 300,
    batch_size: int = 16,
    crop_size: int = 96,
    data_dir: str | None = None,
) -> dict:
    """Get full nnBYOL3D training config."""
    cfg = dict(**NNBYOL3D_CONFIG)
    cfg["num_epochs"] = epochs
    cfg["data"] = dict(**cfg["data"], data_dir=data_dir)
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["crop_size"] = crop_size
    cfg["base_ema"] = 0.99 if epochs <= 300 else 0.996
    return cfg
