"""
privacy/accountant.py
---------------------
Privacy budget accountant for FedDP-POI Phase 3.

Tracks cumulative ε expenditure across federated learning rounds
for each client. Supports basic composition (ε_total = Σ ε_round)
and logs a per-round privacy ledger for thesis reporting.

Usage:
    accountant = PrivacyAccountant(total_epsilon=10.0, delta=1e-5)
    accountant.add_round(epsilon=0.8, round_num=1)
    accountant.add_round(epsilon=0.9, round_num=2)
    print(accountant.total_spent())   # 1.7
    print(accountant.remaining())     # 8.3
    accountant.save('results/federated/privacy_log_client_0.json')
"""

import json
import os
from typing import List, Dict, Optional


class PrivacyAccountant:
    """
    Tracks differential privacy budget across FL rounds.

    Uses basic composition theorem:
        ε_total = Σ ε_i   for i rounds

    This is conservative — tighter bounds are possible with
    advanced composition or RDP (Mironov, 2017), but basic
    composition is sufficient for thesis demonstration.

    Attributes:
        total_epsilon : total privacy budget for this client
        delta         : failure probability
        client_id     : client identifier (for logging)
        ledger        : list of per-round privacy records
    """

    def __init__(
        self,
        total_epsilon: float,
        delta:         float = 1e-5,
        client_id:     int   = -1,
    ):
        if total_epsilon <= 0:
            raise ValueError(f'total_epsilon must be > 0, got {total_epsilon}')

        self.total_epsilon = total_epsilon
        self.delta         = delta
        self.client_id     = client_id
        self.ledger: List[Dict] = []

    def add_round(
        self,
        epsilon:          float,
        round_num:        int,
        noise_multiplier: float = 0.0,
        n_steps:          int   = 0,
    ):
        """
        Record privacy expenditure for one FL round.

        Args:
            epsilon          : ε spent in this round
            round_num        : FL round number (for logging)
            noise_multiplier : σ used (optional, for logging)
            n_steps          : gradient steps taken (optional)
        """
        record = {
            'round':            round_num,
            'epsilon_round':    float(epsilon),
            'epsilon_cumulative': self.total_spent() + float(epsilon),
            'epsilon_remaining':  max(0.0, self.remaining() - float(epsilon)),
            'noise_multiplier': float(noise_multiplier),
            'n_steps':          int(n_steps),
            'budget_exhausted': self.is_exhausted_after(epsilon),
        }
        self.ledger.append(record)

    def total_spent(self) -> float:
        """Total ε consumed across all rounds (basic composition)."""
        return sum(r['epsilon_round'] for r in self.ledger)

    def remaining(self) -> float:
        """Remaining ε budget."""
        return max(0.0, self.total_epsilon - self.total_spent())

    def is_exhausted(self) -> bool:
        """True if total spent ≥ total budget."""
        return self.total_spent() >= self.total_epsilon

    def is_exhausted_after(self, epsilon: float) -> bool:
        return (self.total_spent() + epsilon) >= self.total_epsilon

    def summary(self) -> Dict:
        """Return a summary dict for logging/reporting."""
        return {
            'client_id':      self.client_id,
            'total_budget':   self.total_epsilon,
            'delta':          self.delta,
            'total_spent':    self.total_spent(),
            'remaining':      self.remaining(),
            'n_rounds':       len(self.ledger),
            'exhausted':      self.is_exhausted(),
            'guarantee':      f'({self.total_spent():.4f}, {self.delta:.0e})-DP',
            'ledger':         self.ledger,
        }

    def save(self, path: str):
        """Save privacy log to JSON for thesis reporting."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self):
        """Print a thesis-ready privacy summary."""
        s = self.summary()
        print(f'\n  Privacy Accountant — Client {self.client_id}')
        print(f'  Budget     : ε={self.total_epsilon:.3f}, δ={self.delta:.0e}')
        print(f'  Spent      : ε={s["total_spent"]:.4f}')
        print(f'  Remaining  : ε={s["remaining"]:.4f}')
        print(f'  Guarantee  : {s["guarantee"]}')
        print(f'  Rounds     : {s["n_rounds"]}')

    @classmethod
    def load(cls, path: str) -> 'PrivacyAccountant':
        """Restore accountant from saved JSON (for resumed training)."""
        with open(path) as f:
            data = json.load(f)
        acc = cls(
            total_epsilon = data['total_budget'],
            delta         = data['delta'],
            client_id     = data['client_id'],
        )
        acc.ledger = data.get('ledger', [])
        return acc


def aggregate_client_epsilons(client_epsilons: Dict[int, float]) -> Dict:
    """
    Aggregate per-client ε values for server-side reporting.

    Args:
        client_epsilons : {client_id: epsilon_spent_this_round}

    Returns:
        dict with mean, min, max ε across clients
    """
    if not client_epsilons:
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0}

    values = list(client_epsilons.values())
    return {
        'mean':   sum(values) / len(values),
        'min':    min(values),
        'max':    max(values),
        'values': client_epsilons,
    }
