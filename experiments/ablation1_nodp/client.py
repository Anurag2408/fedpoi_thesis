"""
client.py  (Ablation 1 — No Differential Privacy)
------------------------------------------------------------------
Federated Learning Client WITHOUT DP-SGD.

This ablation removes all differential privacy components to measure
the accuracy cost of DP noise. Standard SGD with gradient clipping
(no Gaussian noise) is used instead.

Changes from Phase 3:
  - All privacy imports removed
  - Standard backward + clip_grad_norm + step (no DP noise)
  - No PrivacyAccountant, no epsilon tracking
  - Results saved to results/ablation1_nodp/

Usage:
    python experiments/ablation1_nodp/client.py <client_id>

Pre-requisites:
    python prepare_sequential_data.py
    python osm/prepare_osm_features.py
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

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from model.lstm_model import build_model
from model.dataset    import get_dataloaders, get_dataset_stats
from evaluate         import evaluate_model


# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'num_venues':   38333,   # total unique venues
    'n_categories': 13,      # OSM super-categories (keep for fair comparison)
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
EVAL_K_VALUES = [5, 10, 20]
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR   = os.path.join(ROOT, 'results', 'ablation1_nodp')
os.makedirs(RESULTS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def _load_category_mapping() -> dict:
    """Load OSM venue→category mapping (Phase 2 artefact)."""
    path = os.path.join(ROOT, 'data', 'osm', 'venue_categories.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


class POILSTMClient(fl.client.NumPyClient):
    """
    Flower client — LSTM + OSM categories — No Differential Privacy.

    This is Ablation 1: identical to Phase 3 except DP-SGD is replaced
    with standard gradient clipping and optimizer step (no Gaussian noise).
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

        # Load OSM venue→category mapping
        self.venue_categories = _load_category_mapping()
        osm_loaded = len(self.venue_categories) > 0
        print(f'  OSM features : {"yes (" + str(len(self.venue_categories)) + " venues)" if osm_loaded else "no — categories will be 0"}')

        self.train_loader, self.test_dataset = get_dataloaders(
            sequences_path   = sequences_path,
            seq_len          = TRAIN_CONFIG['seq_len'],
            batch_size       = TRAIN_CONFIG['batch_size'],
            venue_categories = self.venue_categories,
        )

        stats = get_dataset_stats(sequences_path)
        self.n_checkins      = stats['total_checkins']
        self.n_train_samples = len(self.train_loader.dataset)

        print(f'\n[Client {client_id}] Initialized on {DEVICE}')
        print(f'  Users        : {stats["n_users"]}')
        print(f'  Check-ins    : {self.n_checkins:,}')
        print(f'  Train batches: {len(self.train_loader)}')
        print(f'  Test samples : {len(self.test_dataset)}')
        print(f'  DP           : DISABLED (Ablation 1 — No DP)')

        # ── Build model ───────────────────────────────────────────────────────
        self.model = build_model(MODEL_CONFIG).to(self.device)

        # ── Optimizer & loss ──────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr           = TRAIN_CONFIG['learning_rate'],
            weight_decay = TRAIN_CONFIG['weight_decay'],
        )
        self.criterion = nn.CrossEntropyLoss()

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        """
        Standard local training for LOCAL_EPOCHS (no DP noise).

        Uses gradient clipping (norm=1.0) for training stability but
        does NOT add Gaussian noise — this is the No-DP baseline.
        """
        self.round_num += 1
        self.set_parameters(parameters)
        self.model.train()

        total_loss      = 0.0
        total_batches   = 0
        total_grad_norm = 0.0

        for epoch in range(TRAIN_CONFIG['local_epochs']):
            epoch_loss = 0.0
            for batch in self.train_loader:
                venue_seq, time_seq, cat_seq, targets = batch
                venue_seq = venue_seq.to(self.device)   # (B, seq_len)
                time_seq  = time_seq.to(self.device)    # (B, seq_len)
                cat_seq   = cat_seq.to(self.device)     # (B, seq_len)
                targets   = targets.squeeze(1).to(self.device)  # (B,)

                logits = self.model(venue_seq, time_seq, cat_seq)  # (B, num_venues)
                loss   = self.criterion(logits, targets)

                # Guard against NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'  [Client {self.client_id}] WARNING: NaN/Inf loss, skipping batch')
                    self.optimizer.zero_grad()
                    continue

                # ── Standard SGD step (no DP noise) ───────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
                self.optimizer.step()

                epoch_loss      += loss.item()
                total_grad_norm += grad_norm
                total_batches   += 1

            total_loss += epoch_loss
            avg_epoch_loss = epoch_loss / max(len(self.train_loader), 1)
            print(f'  [Client {self.client_id}] Epoch {epoch+1}/{TRAIN_CONFIG["local_epochs"]}  '
                  f'loss={avg_epoch_loss:.4f}')

        avg_loss      = total_loss      / max(total_batches, 1)
        avg_grad_norm = total_grad_norm / max(total_batches, 1)

        print(f'\n  [Client {self.client_id}] Round {self.round_num} Summary (No DP):')
        print(f'    avg_loss={avg_loss:.4f}  avg_grad_norm={avg_grad_norm:.4f}')

        return (
            self.model.get_parameters(),
            self.n_train_samples,
            {
                'train_loss': float(avg_loss),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate on held-out test sequences."""
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
            use_categories = True,
        )

        p5    = metrics.get('P@5',    0.0)
        r5    = metrics.get('R@5',    0.0)
        ndcg5 = metrics.get('NDCG@5', 0.0)
        f1_5  = metrics.get('F1@5',   0.0)

        print(f'\n[Client {self.client_id}] Evaluation (No DP):')
        print(f'  P@5={p5:.4f}  R@5={r5:.4f}  NDCG@5={ndcg5:.4f}  F1@5={f1_5:.4f}')

        loss = 1.0 - ndcg5   # proxy loss for Flower (lower = better)

        return (
            float(loss),
            len(self.test_dataset),
            {
                'precision': float(p5),
                'recall':    float(r5),
                'ndcg':      float(ndcg5),
                'f1':        float(f1_5),
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
        print('Usage: python experiments/ablation1_nodp/client.py <client_id>')
        sys.exit(1)

    start_client(int(sys.argv[1]))
