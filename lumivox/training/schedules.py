"""LR and EMA schedules for pretraining and fine-tuning."""

from __future__ import annotations

import math

import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Cosine LR schedule with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


class PolyLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Polynomial LR scheduler (nnU-Net style)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
    ):
        def lr_lambda(step: int) -> float:
            return (1 - step / max(max_steps, 1)) ** exponent

        super().__init__(optimizer, lr_lambda)
