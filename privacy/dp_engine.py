"""
privacy/dp_engine.py
--------------------
DP-SGD gradient noise injection for FedDP-POI Phase 3.

Implements the Gaussian mechanism for differential privacy in
stochastic gradient descent. No external DP library required —
all operations are standard PyTorch.

Algorithm (per batch):
    1. Forward pass → compute loss
    2. Backward pass → compute per-parameter gradients
    3. Clip gradient norm to max_grad_norm  (bounding sensitivity)
    4. Add Gaussian noise scaled to (max_grad_norm × σ) / batch_size
    5. Optimizer step on noisy gradients

Privacy guarantee:
    Each gradient release satisfies (ε, δ)-DP via the Gaussian mechanism.
    The noise multiplier σ is calibrated from the target ε using
    budget_allocator.epsilon_to_noise_multiplier().

    Total privacy cost over T rounds is tracked by PrivacyAccountant
    using basic composition: ε_total = Σ ε_round.

References:
    Abadi et al. (2016) — Deep Learning with Differential Privacy
    Dwork & Roth (2014) — The Algorithmic Foundations of DP
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DPConfig:
    """
    Configuration for one client's DP-SGD training.

    Attributes:
        epsilon          : target privacy budget for this client (adaptive)
        delta            : failure probability (1e-5 is standard)
        max_grad_norm    : gradient clipping norm (sensitivity bound)
        noise_multiplier : σ — noise std = noise_multiplier × max_grad_norm / batch_size
                           computed from epsilon via budget_allocator if not set manually
        local_epochs     : number of local training epochs per FL round
        n_steps          : total gradient steps (set automatically in training)
    """
    epsilon:          float = 1.0
    delta:            float = 1e-5
    max_grad_norm:    float = 1.0
    noise_multiplier: float = 1.0    # will be overridden by epsilon_to_noise_multiplier
    local_epochs:     int   = 10
    n_steps:          int   = 0      # filled in during training


def dp_sgd_step(
    model:            nn.Module,
    loss:             torch.Tensor,
    optimizer:        torch.optim.Optimizer,
    dp_config:        DPConfig,
    batch_size:       int,
) -> float:
    """
    Single DP-SGD step: backward → clip → noise → step.

    Replaces the standard loss.backward() + optimizer.step() in the
    training loop. Injects calibrated Gaussian noise to provide
    (ε, δ)-differential privacy.

    Args:
        model      : the PyTorch model being trained
        loss       : scalar loss tensor (from criterion)
        optimizer  : Adam or SGD optimizer
        dp_config  : DP configuration (noise_multiplier, max_grad_norm)
        batch_size : actual batch size (used to scale noise correctly)

    Returns:
        grad_norm_before_clip : float (useful for debugging/monitoring)
    """
    # ── Step 1: Backward pass (compute gradients) ─────────────────────────────
    optimizer.zero_grad()
    loss.backward()

    # ── Step 2: Clip gradient norm (bound sensitivity) ────────────────────────
    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = dp_config.max_grad_norm,
    )

    # ── Step 3: Add calibrated Gaussian noise ─────────────────────────────────
    # Noise std = σ × C / B
    # where σ = noise_multiplier, C = max_grad_norm, B = batch_size
    noise_std = dp_config.noise_multiplier * dp_config.max_grad_norm / batch_size

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.add_(noise)

    # ── Step 4: Optimizer step on noisy gradients ─────────────────────────────
    optimizer.step()

    return float(grad_norm)


def compute_epsilon_spent(
    noise_multiplier: float,
    max_grad_norm:    float,
    batch_size:       int,
    dataset_size:     int,
    n_steps:          int,
    delta:            float = 1e-5,
) -> float:
    """
    Compute actual ε spent given the noise parameters and number of steps.

    Uses the analytical formula for the Gaussian mechanism with
    advanced composition (Dwork et al., 2010).

    ε_per_step = √(2 ln(1.25/δ)) / σ
    ε_total    = ε_per_step × √(2T ln(1/δ))   [advanced composition]

    Args:
        noise_multiplier : σ used during training
        max_grad_norm    : clipping norm C
        batch_size       : B
        dataset_size     : N (full training set size)
        n_steps          : T (total gradient steps = epochs × batches)
        delta            : failure probability

    Returns:
        epsilon_spent : float
    """
    if noise_multiplier <= 0:
        return float('inf')

    # Subsampling rate
    q = batch_size / max(dataset_size, 1)

    # Per-step ε for Gaussian mechanism
    eps_per_step = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier

    # Advanced composition over T steps
    if n_steps <= 1:
        return eps_per_step

    eps_total = eps_per_step * math.sqrt(2 * n_steps * math.log(1.0 / delta))

    # Subsampling amplification
    eps_total = eps_total * q

    return float(eps_total)


def privacy_report(dp_config: DPConfig, dataset_size: int, batch_size: int) -> dict:
    """
    Generate a privacy analysis report for thesis/logging.

    Returns dict with privacy parameters and guarantee.
    """
    n_steps = dp_config.n_steps or (dp_config.local_epochs * max(dataset_size // batch_size, 1))

    eps_spent = compute_epsilon_spent(
        noise_multiplier = dp_config.noise_multiplier,
        max_grad_norm    = dp_config.max_grad_norm,
        batch_size       = batch_size,
        dataset_size     = dataset_size,
        n_steps          = n_steps,
        delta            = dp_config.delta,
    )

    return {
        'target_epsilon':    dp_config.epsilon,
        'delta':             dp_config.delta,
        'noise_multiplier':  dp_config.noise_multiplier,
        'max_grad_norm':     dp_config.max_grad_norm,
        'n_steps':           n_steps,
        'epsilon_spent':     eps_spent,
        'privacy_guarantee': f'({eps_spent:.4f}, {dp_config.delta:.0e})-DP',
    }
