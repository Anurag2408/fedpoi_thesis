"""
model/dataset.py
----------------
PyTorch Dataset for sequential POI recommendation — Phase 2 (with OSM categories).

Returns (venue_seq, time_seq, cat_seq, target) for all modes.
cat_seq is all-zeros when no venue_categories mapping is provided (Phase 1 mode).
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_SEQ_LEN    = 10
DEFAULT_BATCH_SIZE = 64


class POISequenceDataset(Dataset):
    """
    Sliding-window dataset over user check-in sequences.

    Args:
        user_sequences   : list of sequences, each a list of
                           (venue_id_encoded, hour_of_day) tuples
        seq_len          : look-back window size
        mode             : 'train' or 'test'
        venue_categories : optional dict {venue_id_encoded: category_id}
                           if None, category features are all zeros (Phase 1 compat)
    """

    def __init__(
        self,
        user_sequences:   list,
        seq_len:          int  = DEFAULT_SEQ_LEN,
        mode:             str  = 'train',
        venue_categories: dict = None,
    ):
        self.seq_len          = seq_len
        self.mode             = mode
        self.venue_categories = venue_categories   # None = no OSM features
        self.samples          = []
        self.user_train_venues = []

        self._build(user_sequences)

    def _build(self, user_sequences: list):
        for seq in user_sequences:
            if len(seq) < self.seq_len + 1:
                continue

            venue_ids = [s[0] for s in seq]
            time_ids  = [s[1] for s in seq]

            if self.mode == 'train':
                end = len(venue_ids) - 1
            else:
                end = len(venue_ids)

            windows_added = 0
            for i in range(len(venue_ids) - self.seq_len):
                if self.mode == 'train' and i + self.seq_len >= end:
                    break
                if self.mode == 'test' and i + self.seq_len < end - 1:
                    continue

                in_venues = venue_ids[i : i + self.seq_len]
                in_times  = time_ids [i : i + self.seq_len]
                target    = venue_ids[i + self.seq_len]

                if target < 0:
                    continue

                # Build category sequence
                if self.venue_categories is not None:
                    in_cats = [self.venue_categories.get(v, 0) for v in in_venues]
                else:
                    in_cats = [0] * self.seq_len   # zeros = no category info

                self.samples.append((in_venues, in_times, in_cats, target))
                windows_added += 1

            if self.mode == 'test':
                train_venues = set(venue_ids[:-1])
                self.user_train_venues.append(train_venues)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        venues, times, cats, target = self.samples[idx]
        return (
            torch.LongTensor(venues),    # (seq_len,)
            torch.LongTensor(times),     # (seq_len,)
            torch.LongTensor(cats),      # (seq_len,)
            torch.LongTensor([target]),  # (1,)
        )


def get_dataloaders(
    sequences_path:   str,
    seq_len:          int  = DEFAULT_SEQ_LEN,
    batch_size:       int  = DEFAULT_BATCH_SIZE,
    num_workers:      int  = 0,
    venue_categories: dict = None,
) -> tuple:
    """
    Load sequences.pkl and return (train_loader, test_dataset).

    Args:
        sequences_path   : path to sequences.pkl for this client
        seq_len          : look-back window size
        batch_size       : training batch size
        num_workers      : DataLoader workers
        venue_categories : optional {venue_id_encoded: category_id} mapping

    Returns:
        train_loader  : DataLoader
        test_dataset  : POISequenceDataset in test mode
    """
    with open(sequences_path, 'rb') as f:
        user_sequences = pickle.load(f)

    train_dataset = POISequenceDataset(
        user_sequences, seq_len=seq_len, mode='train',
        venue_categories=venue_categories,
    )
    test_dataset = POISequenceDataset(
        user_sequences, seq_len=seq_len, mode='test',
        venue_categories=venue_categories,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = False,
    )

    return train_loader, test_dataset


def get_dataset_stats(sequences_path: str) -> dict:
    with open(sequences_path, 'rb') as f:
        seqs = pickle.load(f)
    total = sum(len(s) for s in seqs)
    lengths = [len(s) for s in seqs]
    return {
        'n_users':        len(seqs),
        'total_checkins': total,
        'avg_seq_len':    total / max(len(seqs), 1),
        'min_seq_len':    min(lengths) if lengths else 0,
        'max_seq_len':    max(lengths) if lengths else 0,
    }
