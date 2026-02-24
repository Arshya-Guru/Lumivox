"""Shared default configuration for all models.

Encoder, optimizer, data, and crop settings that are identical
across simclr and nnbyol3d for fairness.
"""

# Encoder (ResEncL for fair models)
ENCODER_CONFIG = dict(
    name="ResEncL",
    num_input_channels=1,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
)

# Projection head (shared by simclr + nnbyol3d)
PROJECTION_CONFIG = dict(
    input_dim=320,  # ResEncL bottleneck after GAP
    hidden_dim=4096,
    output_dim=256,
)

# Default optimizer: AdamW + Cosine with Warmup
DEFAULT_OPTIMIZER_CONFIG = dict(
    name="adamw",
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    exclude_from_wd=["bias", "norm", "bn"],
)

DEFAULT_SCHEDULE_CONFIG = dict(
    name="cosine",
    warmup_epochs=10,
)

# Data/crop settings
DEFAULT_DATA_CONFIG = dict(
    crop_size=96,
    min_overlap_per_axis=0.5,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
)

# Fine-tuning config (identical for all models)
FINETUNE_CONFIG = dict(
    optimizer="adamw",
    lr=1e-3,
    encoder_lr_factor=0.1,
    weight_decay=1e-2,
    schedule="poly",
    poly_exponent=0.9,
    warmup_epochs=5,
    loss="dice_ce",
    deep_supervision=True,
    deep_supervision_weights=[1.0, 0.5, 0.25, 0.125, 0.0625],
    crop_size=96,
    batch_size=2,
    num_epochs=200,
    precision="bf16-mixed",
)

# Mixed precision default
DEFAULT_PRECISION = "bf16-mixed"
