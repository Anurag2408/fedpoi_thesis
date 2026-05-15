"""
client.py  (Ablation 3 — No OSM Category Features)
------------------------------------------------------------------
Federated Learning Client with DP-SGD but WITHOUT OSM category features.

This ablation removes OpenStreetMap venue categories to measure how
much they contribute to recommendation quality. All DP components
(adaptive ε, DP-SGD, PrivacyAccountant) are retained.

Changes from Phase 3:
  - MODEL_CONFIG: n_categories=0 (disables category embedding)
  - venue_categories set to {} (no OSM loading)
  - get_dataloaders called with venue_categories={}
  - evaluate_model called with use_categories=False
  - Results saved to results/ablation3_noosm/

Usage:
    python experiments/ablation3_noosm/client.py <client_id>

Pre-requisites:
    python prepare_sequential_data.py
    (OSM features not required for this ablation)
"""

# ── gRPC env vars MUST be set before any grpc/flwr import ─────────────────────
import os
os.environ['GRPC_MAX_SEND_MESSAGE_LENGTH']    = str(512 * 1024 * 1024)  # 512 MB
os.environ['GRPC_MAX_RECEIVE_MESSAGE_LENGTH'] = str(512 * 1024 * 1024)
os.environ['GRPC_VERBOSITY']                  = 'ERROR'
# ─────────────────────────────────────────────────────────────────────────────

# ── Repo root on sys.path (allows 'from model...', 'from privacy...' etc.) ──
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from model.lstm_model import build_model
from model.dataset    import get_dataloaders, get_dataset_stats
from evaluate         import evaluate_model

# ── Phase 3: Differential Privacy components ──────────────────────────────────
from privacy import (
    compute_client_epsilon,
    epsilon_to_noise_multiplier,
    dp_sgd_step,
    DPConfig,
    PrivacyAccountant,
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'num_venues':   38333,   # total unique venues
    'n_categories': 0,       # ← 0 disables OSM category embedding (Ablation 3)
    'embed_dim':    64,
    'hidden_dim':   128,
    'num_layers':   2,
    'dropout':      0.3,
}
TRAIN_CONFIG = {
    'seq_len':        10,
    'batch_size':     64,
    'local_epochs':   10,
    'learning_rate':  2e-4,
    'weight_decay':   1e-4,
}
DP_CONFIG_DEFAULTS = {
    'delta':         1e-5,
    'max_grad_norm': 1.0,    # gradient clipping norm (sensitivity bound)
    'eps_min':       0.5,    # ε for client with fewest check-ins
    'eps_max':       2.0,    # ε for client with most check-ins
}
EVAL_K_VALUES = [5, 10, 20]
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR   = os.path.join(ROOT, 'results', 'ablation3_noosm')
os.makedirs(RESULTS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def _load_checkin_counts() -> dict:
    """
    Load the total check-in count for every client from their sequences.pkl.

    Returns:
        {client_id: n_checkins}   for clients 0..9
    """
    counts = {}
    for cid in range(10):
        path = os.path.join(ROOT, 'data', 'federated', f'client_{cid}', 'sequences.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                seqs = pickle.load(f)
            counts[cid] = sum(len(s) for s in seqs)
    return counts


class POILSTMClient(fl.client.NumPyClient):
    """
    Flower client — LSTM + Adaptive DP-SGD — No OSM Category Features.

    This is Ablation 3: identical to Phase 3 except OSM venue categories
    are disabled (n_categories=0, venue_categories={}).
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device    = DEVICE
        self.round_num = 0      # incremented in fit()

        # ── Load data ─────────────────────────────────────────────────────────
        self.client_dir = os.path.join(ROOT, 'data', 'federated', f'client_{client_id}')
        sequences_path  = os.path.join(self.client_dir, 'sequences.pkl')

        if not os.path.exists(sequences_path):
            raise FileNotFoundError(
                f'[Client {client_id}] sequences.pkl not found.\n'
                f'  Run: python prepare_sequential_data.py'
            )

        # Ablation 3: no OSM features — use empty mapping
        self.venue_categories = {}
        print(f'  OSM features : DISABLED (Ablation 3 — No OSM)')

        self.train_loader, self.test_dataset = get_dataloaders(
            sequences_path   = sequences_path,
            seq_len          = TRAIN_CONFIG['seq_len'],
            batch_size       = TRAIN_CONFIG['batch_size'],
            venue_categories = {},
        )

        stats = get_dataset_stats(sequences_path)
        self.n_checkins      = stats['total_checkins']
        self.n_train_samples = len(self.train_loader.dataset)

        print(f'\n[Client {client_id}] Initialized on {DEVICE}')
        print(f'  Users        : {stats["n_users"]}')
        print(f'  Check-ins    : {self.n_checkins:,}')
        print(f'  Train batches: {len(self.train_loader)}')
        print(f'  Test samples : {len(self.test_dataset)}')

        # ── Build model ───────────────────────────────────────────────────────
        self.model = build_model(MODEL_CONFIG).to(self.device)

        # ── Optimizer & loss ──────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr           = TRAIN_CONFIG['learning_rate'],
            weight_decay = TRAIN_CONFIG['weight_decay'],
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── Adaptive DP budget (same as Phase 3) ─────────────────────────────
        all_counts = _load_checkin_counts()
        if not all_counts:
            all_counts = {client_id: self.n_checkins}

        self.epsilon = compute_client_epsilon(
            n_checkins   = self.n_checkins,
            all_checkins = list(all_counts.values()),
            eps_min      = DP_CONFIG_DEFAULTS['eps_min'],
            eps_max      = DP_CONFIG_DEFAULTS['eps_max'],
        )

        n_batches_per_round = max(len(self.train_loader), 1)
        self.n_steps_per_round = TRAIN_CONFIG['local_epochs'] * n_batches_per_round

        sampling_rate = TRAIN_CONFIG['batch_size'] / max(self.n_checkins, 1)

        self.noise_multiplier = epsilon_to_noise_multiplier(
            epsilon       = self.epsilon,
            delta         = DP_CONFIG_DEFAULTS['delta'],
            n_steps       = self.n_steps_per_round,
            sampling_rate = sampling_rate,
        )

        self.dp_config = DPConfig(
            epsilon          = self.epsilon,
            delta            = DP_CONFIG_DEFAULTS['delta'],
            max_grad_norm    = DP_CONFIG_DEFAULTS['max_grad_norm'],
            noise_multiplier = self.noise_multiplier,
            local_epochs     = TRAIN_CONFIG['local_epochs'],
            n_steps          = self.n_steps_per_round,
        )

        self.epsilon_per_round = self.epsilon / 10.0

        self.accountant = PrivacyAccountant(
            total_epsilon = self.epsilon,
            delta         = DP_CONFIG_DEFAULTS['delta'],
            client_id     = client_id,
        )

        print(f'\n[Client {client_id}] DP Configuration (No OSM ablation):')
        print(f'  Adaptive ε   : {self.epsilon:.4f}  '
              f'(range [{DP_CONFIG_DEFAULTS["eps_min"]}, {DP_CONFIG_DEFAULTS["eps_max"]}])')
        print(f'  δ            : {DP_CONFIG_DEFAULTS["delta"]:.0e}')
        print(f'  Noise σ      : {self.noise_multiplier:.4f}')
        print(f'  Clip norm C  : {DP_CONFIG_DEFAULTS["max_grad_norm"]}')
        print(f'  Steps/round  : {self.n_steps_per_round}')
        print(f'  ε/round est. : {self.epsilon_per_round:.4f}')

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        """DP-SGD local training without OSM category features."""
        self.round_num += 1
        self.set_parameters(parameters)
        self.model.train()

        total_loss    = 0.0
        total_batches = 0
        total_grad_norm = 0.0

        for epoch in range(TRAIN_CONFIG['local_epochs']):
            epoch_loss = 0.0
            for batch in self.train_loader:
                venue_seq, time_seq, cat_seq, targets = batch
                venue_seq = venue_seq.to(self.device)   # (B, seq_len)
                time_seq  = time_seq.to(self.device)    # (B, seq_len)
                cat_seq   = cat_seq.to(self.device)     # (B, seq_len) — all zeros
                targets   = targets.squeeze(1).to(self.device)  # (B,)

                logits = self.model(venue_seq, time_seq, cat_seq)  # (B, num_venues)
                loss   = self.criterion(logits, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'  [Client {self.client_id}] WARNING: NaN/Inf loss, skipping batch')
                    self.optimizer.zero_grad()
                    continue

                # ── DP-SGD step: clip → noise → update ────────────────────────
                actual_batch = venue_seq.size(0)
                grad_norm = dp_sgd_step(
                    model      = self.model,
                    loss       = loss,
                    optimizer  = self.optimizer,
                    dp_config  = self.dp_config,
                    batch_size = actual_batch,
                )

                epoch_loss      += loss.item()
                total_grad_norm += grad_norm
                total_batches   += 1

            total_loss += epoch_loss
            avg_epoch_loss = epoch_loss / max(len(self.train_loader), 1)
            print(f'  [Client {self.client_id}] Epoch {epoch+1}/{TRAIN_CONFIG["local_epochs"]}  '
                  f'loss={avg_epoch_loss:.4f}')

        avg_loss      = total_loss      / max(total_batches, 1)
        avg_grad_norm = total_grad_norm / max(total_batches, 1)

        # ── Record privacy expenditure ─────────────────────────────────────────
        self.accountant.add_round(
            epsilon          = self.epsilon_per_round,
            round_num        = self.round_num,
            noise_multiplier = self.noise_multiplier,
            n_steps          = self.n_steps_per_round,
        )
        epsilon_spent = self.accountant.total_spent()
        epsilon_remaining = self.accountant.remaining()

        log_path = os.path.join(
            RESULTS_DIR, f'privacy_log_client_{self.client_id}.json'
        )
        self.accountant.save(log_path)

        print(f'\n  [Client {self.client_id}] Round {self.round_num} DP Summary:')
        print(f'    avg_loss={avg_loss:.4f}  avg_grad_norm={avg_grad_norm:.4f}')
        print(f'    ε_spent={epsilon_spent:.4f}  ε_remaining={epsilon_remaining:.4f}')
        print(f'    guarantee: ({epsilon_spent:.4f}, {DP_CONFIG_DEFAULTS["delta"]:.0e})-DP')

        return (
            self.model.get_parameters(),
            self.n_train_samples,
            {
                'train_loss':       float(avg_loss),
                'epsilon':          float(epsilon_spent),
                'epsilon_round':    float(self.epsilon_per_round),
                'noise_multiplier': float(self.noise_multiplier),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate on held-out test sequences (no OSM categories)."""
        self.set_parameters(parameters)

        if len(self.test_dataset) == 0:
            print(f'  [Client {self.client_id}] No test data.')
            return 0.0, 1, {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'f1': 0.0}

        metrics = evaluate_model(
            model          = self.model,
            test_dataset   = self.test_dataset,
            num_venues     = MODEL_CONFIG['num_venues'],
            k_values       = EVAL_K_VALUES,
            device         = self.device,
            mask_mode      = 'window',
            use_categories = False,  # ← Ablation 3: no OSM categories
        )

        p5    = metrics.get('P@5',    0.0)
        r5    = metrics.get('R@5',    0.0)
        ndcg5 = metrics.get('NDCG@5', 0.0)
        f1_5  = metrics.get('F1@5',   0.0)

        epsilon_spent = self.accountant.total_spent()

        print(f'\n[Client {self.client_id}] Evaluation (No OSM):')
        print(f'  P@5={p5:.4f}  R@5={r5:.4f}  NDCG@5={ndcg5:.4f}  F1@5={f1_5:.4f}')
        print(f'  Privacy: ε_spent={epsilon_spent:.4f}  '
              f'(target ε={self.epsilon:.4f})')

        loss = 1.0 - ndcg5   # proxy loss for Flower (lower = better)

        return (
            float(loss),
            len(self.test_dataset),
            {
                'precision': float(p5),
                'recall':    float(r5),
                'ndcg':      float(ndcg5),
                'f1':        float(f1_5),
                'epsilon':   float(epsilon_spent),
            },
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def start_client(client_id: int):
    client = POILSTMClient(client_id=client_id)
    fl.client.start_numpy_client(
        server_address          = 'localhost:8080',
        client                  = client,
        grpc_max_message_length = 512 * 1024 * 1024,
    )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python experiments/ablation3_noosm/client.py <client_id>')
        sys.exit(1)

    start_client(int(sys.argv[1]))
