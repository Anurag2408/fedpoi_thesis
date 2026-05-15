"""
evaluate.py  (Phase 2 — supports optional OSM category sequences)
-----------
Evaluation metrics for POI recommendation.

Metrics:
    Precision@K, Recall@K, NDCG@K, F1@K, Coverage@K
"""

import math
import numpy as np
import torch
from torch.utils.data import DataLoader


# ── Core metric functions ─────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return sum(1 for v in recommended[:k] if v in relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for v in recommended[:k] if v in relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg   = sum(1.0 / math.log2(i + 2) for i, v in enumerate(top_k) if v in relevant)
    idcg  = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def coverage_at_k(all_recommended: list, num_venues: int) -> float:
    unique = set(v for recs in all_recommended for v in recs)
    return len(unique) / num_venues if num_venues > 0 else 0.0


# ── Full evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(
    model,
    test_dataset,
    num_venues:      int,
    k_values:        list = [5, 10, 20],
    device:          str  = 'cpu',
    mask_mode:       str  = 'window',   # 'window' | 'history' | 'none'
    use_categories:  bool = False,
) -> dict:
    """
    Run full evaluation.

    Args:
        model           : POILSTMModel
        test_dataset    : POISequenceDataset in test mode
        num_venues      : total unique venues
        k_values        : list of K cutoffs
        device          : 'cpu' or 'cuda'
        mask_mode       : venue exclusion strategy
                          'window'  — exclude input window only (default, NMF-comparable)
                          'history' — exclude full training history
                          'none'    — no exclusion
        use_categories  : whether to pass cat_seq to the model (Phase 2)

    Returns:
        dict of metric_name → float
    """
    model.eval()
    model.to(device)

    max_k = max(k_values)

    results = {f'P@{k}':    [] for k in k_values}
    results.update({f'R@{k}':    [] for k in k_values})
    results.update({f'NDCG@{k}': [] for k in k_values})
    results.update({f'F1@{k}':   [] for k in k_values})
    all_recommended = []

    has_train_venues = len(test_dataset.user_train_venues) == len(test_dataset)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Dataset returns 4-tuple in Phase 2
            venue_seq, time_seq, cat_seq, target = test_dataset[idx]

            target_venue = target.item()
            relevant     = {target_venue}

            # Build exclusion set
            if mask_mode == 'none':
                exclude_set = set()
            elif mask_mode == 'window':
                exclude_set = set(venue_seq.tolist())
                exclude_set.discard(target_venue)
            else:  # 'history'
                if has_train_venues:
                    exclude_set = {v for v in test_dataset.user_train_venues[idx]
                                   if v != target_venue}
                else:
                    exclude_set = set()

            # Forward pass
            v_in = venue_seq.unsqueeze(0).to(device)
            t_in = time_seq.unsqueeze(0).to(device)
            c_in = cat_seq.unsqueeze(0).to(device) if use_categories else None

            logits = model(v_in, t_in, c_in).squeeze(0)

            # Mask excluded venues
            if exclude_set:
                mask_idx = [v for v in exclude_set if 0 <= v < logits.size(0)]
                if mask_idx:
                    logits[mask_idx] = float('-inf')

            top_preds = torch.topk(logits, min(max_k, logits.size(0))).indices.tolist()
            all_recommended.append(top_preds[:max_k])

            for k in k_values:
                results[f'P@{k}'].append(precision_at_k(top_preds, relevant, k))
                results[f'R@{k}'].append(recall_at_k(top_preds, relevant, k))
                results[f'NDCG@{k}'].append(ndcg_at_k(top_preds, relevant, k))
                results[f'F1@{k}'].append(f1_at_k(top_preds, relevant, k))

    avg_results = {key: float(np.mean(vals)) for key, vals in results.items()}

    for k in k_values:
        avg_results[f'Cov@{k}'] = coverage_at_k(
            [r[:k] for r in all_recommended], num_venues
        )

    return avg_results


def print_results(results: dict, title: str = 'Evaluation Results'):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for metric, value in sorted(results.items()):
        bar = '█' * int(value * 200)
        print(f"  {metric:<12}: {value:.4f}  {bar}")
    print(f"{'='*55}\n")
