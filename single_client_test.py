import numpy as np
from sklearn.decomposition import NMF

# Load client 0 data
client_id = 0
train_matrix = np.load(f'data/federated/client_{client_id}/train_matrix.npy')
test_matrix = np.load(f'data/federated/client_{client_id}/test_matrix.npy')

print(f"Client {client_id} Data:")
print(f"  Train shape: {train_matrix.shape}")
print(f"  Test shape: {test_matrix.shape}")
print(f"  Train interactions: {train_matrix.sum()}")
print(f"  Test interactions: {test_matrix.sum()}")
print(f"  Test users with data: {(test_matrix.sum(axis=1) > 0).sum()}")

# Train a model
print("\nTraining model...")
model = NMF(n_components=10, init='random', random_state=42, max_iter=50, verbose=0)
user_features = model.fit_transform(train_matrix)
venue_features = model.components_

print(f"  User features shape: {user_features.shape}")
print(f"  Venue features shape: {venue_features.shape}")


# Make predictions
predicted = np.dot(user_features, venue_features)
print(f"  Predictions shape: {predicted.shape}")

# Calculate Precision@5
print("\nCalculating Precision@5...")
precisions = []
for user_idx in range(predicted.shape[0]):
    actual_venues = set(np.where(test_matrix[user_idx] == 1)[0])
    if len(actual_venues) == 0:
        continue

    train_venues = set(np.where(train_matrix[user_idx] == 1)[0])
    scores = predicted[user_idx].copy()
    scores[list(train_venues)] = -np.inf

    top_5 = np.argsort(scores)[-5:][::-1]
    hits = len(set(top_5) & actual_venues)
    precisions.append(hits / 5)
print(f"  Evaluated users: {len(precisions)} out of {predicted.shape[0]}")

if precisions:
    avg_precision = np.mean(precisions)
    print(f"\n✓ SUCCESS!")
    print(f"  Precision@5: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"\nThis proves metric calculation works!")
else:
    print("\n✗ FAILED!")
    print("  No valid users for evaluation")
    print("  All users might have empty test sets")


# Additional debugging
print("\n" + "="*60)
print("DETAILED DEBUG INFO:")
print("="*60)

for user_idx in range(min(5, predicted.shape[0])):
    train_count = train_matrix[user_idx].sum()
    test_count = test_matrix[user_idx].sum()
    print(f"User {user_idx}: Train={train_count:.0f}, Test={test_count:.0f}")