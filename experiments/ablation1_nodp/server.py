"""
server.py  (Ablation 1 — No Differential Privacy)
------------------------------------------------------------------
Federated Learning Server — No DP Ablation

Identical to Phase 3 server but saves results to ablation1_nodp/.
No epsilon aggregation since clients do not report privacy metrics.
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
import numpy as np
from typing import List, Tuple, Optional, Dict

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR    = os.path.join(ROOT, 'results', 'ablation1_nodp')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def weighted_average_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate evaluation metrics from all clients using weighted average."""
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    def wavg(key):
        return sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples

    return {
        'precision': wavg('precision'),
        'recall':    wavg('recall'),
        'ndcg':      wavg('ndcg'),
        'f1':        wavg('f1'),
    }


def save_checkpoint(parameters, round_num: int):
    """Save aggregated model weights to disk after each round."""
    ndarrays = parameters_to_ndarrays(parameters)
    path = os.path.join(CHECKPOINT_DIR, f'round_{round_num:03d}.npz')
    np.savez(path, *ndarrays)
    latest = os.path.join(CHECKPOINT_DIR, 'latest.npz')
    np.savez(latest, *ndarrays)
    print(f'  [Checkpoint] Saved → {path}')


def load_checkpoint(path: str = None):
    """Load model weights from a checkpoint file."""
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, 'latest.npz')

    if not os.path.exists(path):
        return None

    data = np.load(path)
    ndarrays = [data[k] for k in sorted(data.files)]
    print(f'  [Checkpoint] Loaded → {path}')
    return ndarrays_to_parameters(ndarrays)


def load_metrics_history() -> List[Dict]:
    """Load existing metrics.json so resumed runs append to it."""
    path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy for Ablation 1 (No DP)."""

    def __init__(self, resume: bool = False, **kwargs):
        kwargs['evaluate_metrics_aggregation_fn'] = weighted_average_metrics

        if resume:
            ckpt = load_checkpoint()
            if ckpt is not None:
                kwargs['initial_parameters'] = ckpt
                print('  [Resume] Starting from saved checkpoint.')
            else:
                print('  [Resume] No checkpoint found — starting from scratch.')

        super().__init__(**kwargs)
        self.round_metrics = load_metrics_history() if resume else []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            print(f'\n[Server] Round {server_round} — aggregated {len(results)} clients')
            save_checkpoint(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      List,
        failures:     List,
    ) -> Tuple[Optional[float], Dict]:

        if not results:
            return 0.0, {}

        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)

        if loss_agg is None:
            loss_agg = 0.0
        if metrics_agg is None:
            metrics_agg = {}

        p5    = metrics_agg.get('precision', 0.0)
        r5    = metrics_agg.get('recall',    0.0)
        ndcg5 = metrics_agg.get('ndcg',      0.0)
        f1_5  = metrics_agg.get('f1',        0.0)

        print(f'\n[Server] Round {server_round} Results (No DP):')
        print(f'  P@5={p5*100:.2f}%  R@5={r5*100:.2f}%  '
              f'NDCG@5={ndcg5*100:.2f}%  F1@5={f1_5*100:.2f}%')

        round_data = {
            'round':     server_round,
            'loss':      float(loss_agg),
            'precision': float(p5),
            'recall':    float(r5),
            'ndcg':      float(ndcg5),
            'f1':        float(f1_5),
            'n_clients': len(results),
        }
        self.round_metrics.append(round_data)

        out_path = os.path.join(RESULTS_DIR, 'metrics.json')
        with open(out_path, 'w') as f:
            json.dump(self.round_metrics, f, indent=2)

        return loss_agg, metrics_agg

    def print_summary(self):
        """Print a summary table of all rounds."""
        if not self.round_metrics:
            return

        print('\n' + '='*65)
        print(f'  {"Round":>5}  {"P@5":>8}  {"R@5":>8}  {"NDCG@5":>8}  {"F1@5":>8}')
        print('  ' + '-'*61)
        for r in self.round_metrics:
            print(f'  {r["round"]:>5}  '
                  f'{r["precision"]*100:>7.2f}%  '
                  f'{r["recall"]*100:>7.2f}%  '
                  f'{r["ndcg"]*100:>7.2f}%  '
                  f'{r["f1"]*100:>7.2f}%')
        print('='*65)

        best = max(self.round_metrics, key=lambda x: x['ndcg'])
        print(f'\n  Best Round {best["round"]}:  '
              f'P@5={best["precision"]*100:.2f}%  '
              f'NDCG@5={best["ndcg"]*100:.2f}%')
        print(f'  Results saved → {RESULTS_DIR}/metrics.json\n')


def start_server(n_rounds: int = 10, n_clients: int = 10, resume: bool = False):
    print('='*60)
    print('  FEDERATED POI RECOMMENDATION — ABLATION 1 (NO DP)')
    print('='*60)
    print(f'\n  Rounds  : {n_rounds}')
    print(f'  Clients : {n_clients}')
    print(f'  Resume  : {resume}')
    print(f'  Waiting for {n_clients} clients to connect ...\n')

    min_clients = max(1, n_clients - 1)

    strategy = SaveModelStrategy(
        resume                = resume,
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = min_clients,
        min_evaluate_clients  = min_clients,
        min_available_clients = n_clients,
    )

    fl.server.start_server(
        server_address          = 'localhost:8080',
        config                  = fl.server.ServerConfig(num_rounds=n_rounds),
        strategy                = strategy,
        grpc_max_message_length = 512 * 1024 * 1024,
    )

    strategy.print_summary()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rounds',  type=int,  default=10)
    p.add_argument('--clients', type=int,  default=10)
    p.add_argument('--resume',  action='store_true',
                   help='Resume from last saved checkpoint')
    args = p.parse_args()
    start_server(n_rounds=args.rounds, n_clients=args.clients, resume=args.resume)
