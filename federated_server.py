"""
federated_server.py
FIXED VERSION - Metrics Aggregation Working
"""

import flwr as fl
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import os


def weighted_average_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate evaluation metrics using weighted average"""

    print(f"\n[Server] === AGGREGATING METRICS ===")
    print(f"[Server] Received metrics from {len(metrics)} clients")

    if not metrics:
        print("[Server] WARNING: No metrics received!")
        return {}

    # Debug: Print what we received from each client
    for i, (num_examples, m) in enumerate(metrics):
        print(f"  Client {i}: {num_examples} examples, P@5={m.get('precision', 0):.4f}, R@5={m.get('recall', 0):.4f}")

    # Calculate weighted averages
    total_examples = sum(num_examples for num_examples, _ in metrics)
    print(f"[Server] Total examples: {total_examples}")

    if total_examples == 0:
        print("[Server] ERROR: Total examples is 0!")
        return {"precision": 0.0, "recall": 0.0}

    # Weighted precision
    weighted_precision = sum(
        num_examples * m.get("precision", 0.0)
        for num_examples, m in metrics
    ) / total_examples

    # Weighted recall
    weighted_recall = sum(
        num_examples * m.get("recall", 0.0)
        for num_examples, m in metrics
    ) / total_examples

    print(f"[Server] Aggregated Precision@5: {weighted_precision:.6f}")
    print(f"[Server] Aggregated Recall@5: {weighted_recall:.6f}")
    print(f"[Server] === AGGREGATION COMPLETE ===\n")

    return {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall)
    }


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        # CRITICAL: Set the aggregation function BEFORE calling super().__init__
        kwargs['evaluate_metrics_aggregation_fn'] = weighted_average_metrics
        super().__init__(**kwargs)
        self.round_metrics = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"\n[Server] Round {server_round} aggregation complete")
            print(f"[Server] Received updates from {len(results)} clients")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List,
            failures: List,
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results from clients"""

        if not results:
            return 0.0, {}

        # Call parent's aggregate_evaluate (which will use our custom aggregation function)
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        if aggregated_result is None:
            return 0.0, {}

        loss_aggregated, metrics_aggregated = aggregated_result

        if loss_aggregated is None:
            loss_aggregated = 0.0

        if metrics_aggregated is None:
            metrics_aggregated = {}

        # Extract metrics
        precision = metrics_aggregated.get('precision', 0)
        recall = metrics_aggregated.get('recall', 0)

        # Save metrics
        round_data = {
            'round': server_round,
            'loss': float(loss_aggregated),
            'precision': float(precision),
            'recall': float(recall),
            'n_clients': len(results)
        }

        self.round_metrics.append(round_data)

        # Save to file
        os.makedirs('results/federated', exist_ok=True)
        with open('results/federated/metrics.json', 'w') as f:
            json.dump(self.round_metrics, f, indent=2)

        # Print metrics
        print(f"\n[Server] Round {server_round} Evaluation:")
        print(f"  Loss:        {loss_aggregated:.6f}")
        print(f"  Precision@5: {precision * 100:.2f}%")
        print(f"  Recall@5:    {recall * 100:.2f}%")

        return loss_aggregated, metrics_aggregated


def start_server(n_rounds=10, n_clients=10):
    print("=" * 60)
    print("FEDERATED POI RECOMMENDATION - SERVER")
    print("=" * 60)
    print(f"\nWaiting for {n_clients} clients to connect...")

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        # evaluate_metrics_aggregation_fn is set in __init__
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        grpc_max_message_length=104857600
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if strategy.round_metrics:
        final_round = strategy.round_metrics[-1]
        print(f"\nFinal Results (Round {final_round['round']}):")
        print(f"  Loss:        {final_round['loss']:.6f}")
        print(f"  Precision@5: {final_round['precision'] * 100:.2f}%")
        print(f"  Recall@5:    {final_round['recall'] * 100:.2f}%")
        print(f"\n✓ Metrics saved to: results/federated/metrics.json")

    print("\n" + "=" * 60)
    print("SERVER COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    start_server(n_rounds=10, n_clients=10)