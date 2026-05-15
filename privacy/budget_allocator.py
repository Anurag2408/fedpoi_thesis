"""
privacy/budget_allocator.py
---------------------------
Adaptive differential privacy budget allocation for FedDP-POI.

Core idea (thesis contribution):
    Instead of assigning the same privacy budget ε to all clients,
    we adaptively scale ε based on each client's local data size.

    Rationale:
      - Users with few check-ins are more privacy-sensitive (each
        check-in reveals a larger fraction of their total mobility).
      - Users with many check-ins contribute more gradient signal,
        so a slightly relaxed privacy budget has lower marginal risk.

    Allocation formula (linear scaling):
        ε_i = ε_min + (ε_max − ε_min) × (n_i − n_min) / (n_max − n_min)

    where n_i is the number of local check-ins for client i.

References:
    Dwork et al. (2006) — Calibrating noise to sensitivity
    Mironov (2017)      — Rényi Differential Privacy
    Geyer et al. (2017) — Client-level DP in federated learning
"""

import math
from typing import Dict, List


# ── Default privacy budget bounds ─────────────────────────────────────────────
EPS_MIN     = 0.5    # strongest privacy  (fewest check-ins)
EPS_MAX     = 2.0    # weakest privacy    (most check-ins)
DELTA       = 1e-5   # failure probability (standard choice for large datasets)
# ─────────────────────────────────────────────────────────────────────────────


def compute_client_epsilon(
    n_checkins:     int,
    all_checkins:   List[int],
    eps_min:        float = EPS_MIN,
    eps_max:        float = EPS_MAX,
) -> float:
    """
    Compute adaptive ε for a single client using linear interpolation.

    Args:
        n_checkins   : number of local check-ins for this client
        all_checkins : list of check-in counts for ALL clients
                       (used to determine the min/max range)
        eps_min      : ε for the client with fewest check-ins
        eps_max      : ε for the client with most check-ins

    Returns:
        epsilon : float in [eps_min, eps_max]

    Example:
        Client has 18,000 check-ins; range is [15,000 – 25,000]
        ratio = (18000 − 15000) / (25000 − 15000) = 0.3
        ε = 0.5 + 0.3 × (2.0 − 0.5) = 0.95
    """
    n_min = min(all_checkins)
    n_max = max(all_checkins)

    if n_max == n_min:
        # All clients identical data size → use midpoint
        return (eps_min + eps_max) / 2.0

    ratio   = (n_checkins - n_min) / (n_max - n_min)
    epsilon = eps_min + ratio * (eps_max - eps_min)
    return float(epsilon)


def allocate_budgets(
    client_checkins: Dict[int, int],
    eps_min:         float = EPS_MIN,
    eps_max:         float = EPS_MAX,
) -> Dict[int, float]:
    """
    Compute ε for every client given a {client_id: n_checkins} mapping.

    Args:
        client_checkins : {client_id: n_checkins}
        eps_min         : minimum ε (fewest check-ins)
        eps_max         : maximum ε (most check-ins)

    Returns:
        {client_id: epsilon}
    """
    all_counts = list(client_checkins.values())
    return {
        cid: compute_client_epsilon(n, all_counts, eps_min, eps_max)
        for cid, n in client_checkins.items()
    }


def epsilon_to_noise_multiplier(
    epsilon:      float,
    delta:        float = DELTA,
    n_steps:      int   = 1,
    sampling_rate: float = 1.0,
) -> float:
    """
    Convert a target (ε, δ) privacy budget to a Gaussian noise multiplier σ.

    Uses the analytical Gaussian mechanism formula with advanced composition
    over n_steps rounds (Dwork et al., 2010 — Boosting and Differential Privacy).

    Single-step Gaussian mechanism:
        σ_single = √(2 ln(1.25/δ)) / ε

    With composition over T steps (strong composition theorem):
        ε_total ≈ ε_per_step × √(2T ln(1/δ))
        ⟹ ε_per_step = ε_total / √(2T ln(1/δ))
        ⟹ σ = √(2 ln(1.25/δ)) / ε_per_step

    Args:
        epsilon       : target total privacy budget
        delta         : failure probability (default 1e-5)
        n_steps       : total number of gradient steps (local_epochs × n_batches)
        sampling_rate : batch sampling rate q = batch_size / dataset_size
                        (used for subsampled Gaussian mechanism amplification)

    Returns:
        noise_multiplier σ (multiply by max_grad_norm to get actual noise std)
    """
    if epsilon <= 0:
        raise ValueError(f'epsilon must be > 0, got {epsilon}')
    if delta <= 0 or delta >= 1:
        raise ValueError(f'delta must be in (0,1), got {delta}')

    if n_steps <= 1:
        # Single step: basic Gaussian mechanism
        sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    else:
        # Advanced composition over T steps
        # ε_per_step = ε / √(2T ln(1/δ))
        eps_per_step = epsilon / math.sqrt(2 * n_steps * math.log(1.0 / delta))
        sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps_per_step

    # Subsampling amplification: σ_effective = σ / q
    # (privacy is amplified when only a fraction q of data is used per step)
    if sampling_rate < 1.0 and sampling_rate > 0:
        sigma = sigma * sampling_rate   # tighter bound with subsampling

    # Clamp to reasonable range
    sigma = max(0.01, min(sigma, 10.0))

    return float(sigma)


def print_budget_table(client_epsilons: Dict[int, float], client_checkins: Dict[int, int]):
    """Print a thesis-ready table of adaptive ε allocations."""
    print('\n  Adaptive ε Budget Allocation:')
    print(f'  {"Client":>8}  {"Check-ins":>10}  {"ε":>6}  {"Noise σ":>8}')
    print('  ' + '-' * 40)
    for cid in sorted(client_epsilons.keys()):
        eps  = client_epsilons[cid]
        n    = client_checkins.get(cid, 0)
        sigma = epsilon_to_noise_multiplier(eps)
        print(f'  {cid:>8}  {n:>10,}  {eps:>6.3f}  {sigma:>8.4f}')
    print()
