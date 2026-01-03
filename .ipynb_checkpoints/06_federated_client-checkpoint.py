"""
06_federated_client.py
Federated Learning Client
"""

import flwr as fl
import numpy as np
from sklearn.decomposition import NMF
import sys
import os

# Set gRPC limits
os.environ['GRPC_MAX_SEND_MESSAGE_LENGTH'] = str(100 * 1024 * 1024)
os.environ['GRPC_MAX_RECEIVE_MESSAGE_LENGTH'] = str(100 * 1024 * 1024)


class POIClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, n_components: int = 10):
        self.client_id = client_id
        self.n_components = n_components

        client_dir = f'data/federated/client_{client_id}'
        self.train_matrix = np.load(f'{client_dir}/train_matrix.npy')
        self.test_matrix = np.load(f'{client_dir}/test_matrix.npy')

        print(f"[Client {client_id}] Initialized")
        print(f"  Train: {self.train_matrix.sum():.0f} interactions")
        print(f"  Test:  {self.test_matrix.sum():.0f} interactions")

        self.model = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            max_iter=50,
            verbose=0
        )

        self.user_features = None
        self.venue_features = None

    def get_parameters(self, config):
        return [self.venue_features] if self.venue_features is not None else []

    def set_parameters(self, parameters):
        if len(parameters) > 0:
            self.venue_features = parameters[0]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if self.venue_features is None:
            self.user_features = self.model.fit_transform(self.train_matrix)
            self.venue_features = self.model.components_
        else:
            self.model.components_ = self.venue_features
            self.user_features = self.model.transform(self.train_matrix)

        return [self.venue_features], len(self.train_matrix), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test data"""

        print(f"\n[Client {self.client_id}] === EVALUATE START ===")

        self.set_parameters(parameters)

        if self.venue_features is None or self.user_features is None:
            print(f"[Client {self.client_id}] ERROR: No trained model!")
            return 0.0, 1, {"precision": 0.0, "recall": 0.0}

        test_interactions = int(self.test_matrix.sum())
        print(f"[Client {self.client_id}] Test interactions: {test_interactions}")

        if test_interactions == 0:
            print(f"[Client {self.client_id}] ERROR: No test data!")
            return 0.0, 1, {"precision": 0.0, "recall": 0.0}

        predicted = np.dot(self.user_features, self.venue_features)

        precision = self._precision_at_k(predicted, self.test_matrix, k=5)
        recall = self._recall_at_k(predicted, self.test_matrix, k=5)
        loss = self._calculate_loss(predicted, self.test_matrix)

        # Convert to Python floats
        precision_float = float(precision)
        recall_float = float(recall)
        loss_float = float(loss)

        print(f"[Client {self.client_id}] Metrics:")
        print(f"  Precision@5: {precision_float:.6f}")
        print(f"  Recall@5: {recall_float:.6f}")
        print(f"  Loss: {loss_float:.6f}")

        metrics_dict = {
            "precision": precision_float,
            "recall": recall_float
        }

        print(f"[Client {self.client_id}] Returning: {metrics_dict}")
        print(f"[Client {self.client_id}] === EVALUATE END ===\n")

        return loss_float, test_interactions, metrics_dict

    def _precision_at_k(self, predicted, actual, k=5):
        precisions = []
        for user_idx in range(predicted.shape[0]):
            actual_venues = set(np.where(actual[user_idx] == 1)[0])
            if len(actual_venues) == 0:
                continue

            train_venues = set(np.where(self.train_matrix[user_idx] == 1)[0])
            scores = predicted[user_idx].copy()
            scores[list(train_venues)] = -np.inf

            top_k = np.argsort(scores)[-k:][::-1]
            hits = len(set(top_k) & actual_venues)
            precisions.append(hits / k)

        return np.mean(precisions) if precisions else 0.0

    def _recall_at_k(self, predicted, actual, k=5):
        recalls = []
        for user_idx in range(predicted.shape[0]):
            actual_venues = set(np.where(actual[user_idx] == 1)[0])
            if len(actual_venues) == 0:
                continue

            train_venues = set(np.where(self.train_matrix[user_idx] == 1)[0])
            scores = predicted[user_idx].copy()
            scores[list(train_venues)] = -np.inf

            top_k = np.argsort(scores)[-k:][::-1]
            hits = len(set(top_k) & actual_venues)
            recalls.append(hits / len(actual_venues))

        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_loss(self, predicted, actual):
        mask = (actual > 0)
        if mask.sum() == 0:
            return 0.0
        return float(np.sum((predicted[mask] - actual[mask]) ** 2) / mask.sum())


def start_client(client_id: int):
    client = POIClient(client_id=client_id, n_components=10)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        grpc_max_message_length=104857600
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 06_federated_client.py <client_id>")
        sys.exit(1)

    start_client(int(sys.argv[1]))
