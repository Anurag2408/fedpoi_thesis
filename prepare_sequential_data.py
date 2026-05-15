"""
prepare_sequential_data.py
--------------------------
Converts the Foursquare check-in data into per-client sequential datasets
for LSTM training. Run this ONCE before starting federated training.

Usage:
    python prepare_sequential_data.py

Output:
    data/federated/client_{id}/sequences.pkl  for each client
    Each sequences.pkl contains a list of user sequences:
        [ [(venue_id_enc, hour), (venue_id_enc, hour), ...],  <- user 0
          [(venue_id_enc, hour), ...],                         <- user 1
          ...
        ]
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_CSV   = 'data/processed/foursquare_filtered.csv'
FEDERATED_DIR   = 'data/federated'
N_CLIENTS       = 10
MIN_SEQ_LEN     = 5          # drop users with fewer than 5 check-ins
TIMESTAMP_FMT   = '%a %b %d %H:%M:%S +0000 %Y'   # e.g. "Tue Apr 03 18:00:09 +0000 2012"
# ──────────────────────────────────────────────────────────────────────────────


def parse_hour(ts_str: str) -> int:
    """Extract hour-of-day (0–23) from Foursquare timestamp string."""
    try:
        dt = datetime.strptime(ts_str.strip(), TIMESTAMP_FMT)
        return dt.hour
    except ValueError:
        return 0


def build_user_sequences(df: pd.DataFrame) -> dict:
    """
    Build {user_id_encoded -> [(venue_id_encoded, hour), ...]} mapping.
    Sequences are sorted chronologically per user.
    """
    df = df.copy()
    df['hour'] = df['utcTimestamp'].apply(parse_hour)
    df = df.sort_values(['user_id_encoded', 'utcTimestamp'])

    sequences = {}
    for uid, group in df.groupby('user_id_encoded'):
        seq = list(zip(
            group['venue_id_encoded'].tolist(),
            group['hour'].tolist()
        ))
        if len(seq) >= MIN_SEQ_LEN:
            sequences[uid] = seq
    return sequences


def main():
    print("=" * 60)
    print("PREPARE SEQUENTIAL DATA")
    print("=" * 60)

    # ── Load processed CSV ────────────────────────────────────────
    print(f"\n[1/3] Loading {PROCESSED_CSV} ...")
    df = pd.read_csv(PROCESSED_CSV)
    print(f"      Loaded {len(df):,} check-ins for {df['user_id_encoded'].nunique()} users")

    # ── Build full user→sequence map ──────────────────────────────
    print(f"\n[2/3] Building chronological user sequences ...")
    all_sequences = build_user_sequences(df)
    print(f"      {len(all_sequences)} users have >= {MIN_SEQ_LEN} check-ins")

    # ── Write per-client sequence files ──────────────────────────
    print(f"\n[3/3] Writing per-client sequence files ...")
    total_seqs = 0

    for client_id in range(N_CLIENTS):
        client_dir = os.path.join(FEDERATED_DIR, f'client_{client_id}')
        user_ids_path = os.path.join(client_dir, 'user_ids.npy')

        if not os.path.exists(user_ids_path):
            print(f"  [!] client_{client_id}: user_ids.npy not found, skipping")
            continue

        client_user_ids = np.load(user_ids_path)   # encoded user IDs

        # Collect sequences for users in this client
        client_sequences = []
        for uid in client_user_ids:
            if uid in all_sequences:
                client_sequences.append(all_sequences[uid])

        # Save
        out_path = os.path.join(client_dir, 'sequences.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(client_sequences, f)

        n_checkins = sum(len(s) for s in client_sequences)
        total_seqs += len(client_sequences)
        print(f"  client_{client_id}: {len(client_sequences)} users, "
              f"{n_checkins:,} check-ins → {out_path}")

    print(f"\n✓ Done. {total_seqs} user sequences written across {N_CLIENTS} clients.")
    print("  You can now run the federated LSTM training.\n")


if __name__ == '__main__':
    main()
