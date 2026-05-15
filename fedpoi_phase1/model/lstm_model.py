"""
model/lstm_model.py
-------------------
Federated LSTM with Self-Attention for next-POI recommendation.

Phase 2 addition: optional OSM category embedding fused into LSTM input.

Architecture:
    POI Embedding      ─┐
    Time Embedding     ─┼─► concat ─► LSTM ─► Self-Attention ─► Linear ─► logits
    Category Embedding ─┘  (category only present if n_categories > 0)

The full model state_dict is shared via FedAvg — only weights are transmitted,
never raw location data.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class SelfAttention(nn.Module):
    """
    Additive self-attention over LSTM hidden states.
    Learns to weight which time-steps are most relevant for prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
        """
        scores  = self.attn(lstm_out)               # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)    # (batch, hidden_dim)
        return context


class POILSTMModel(nn.Module):
    """
    Core recommendation model trained locally on each federated client.

    Parameters
    ----------
    num_venues   : int   total number of unique POIs
    n_categories : int   number of OSM super-categories (0 = no category feature)
    num_hours    : int   24 (hour-of-day embedding)
    embed_dim    : int   POI embedding dimension  (default 64)
    hidden_dim   : int   LSTM hidden size          (default 128)
    num_layers   : int   stacked LSTM layers       (default 2)
    dropout      : float dropout rate              (default 0.3)
    """

    def __init__(
        self,
        num_venues:   int,
        n_categories: int   = 0,      # 0 = Phase 1 mode (no OSM)
        num_hours:    int   = 24,
        embed_dim:    int   = 64,
        hidden_dim:   int   = 128,
        num_layers:   int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.num_venues   = num_venues
        self.embed_dim    = embed_dim
        self.hidden_dim   = hidden_dim
        self.n_categories = n_categories
        self.use_category = n_categories > 0

        cat_dim = embed_dim // 4   # category embedding dimension

        # +1 for padding index 0
        self.venue_embed = nn.Embedding(num_venues + 1, embed_dim, padding_idx=0)
        self.time_embed  = nn.Embedding(num_hours,      embed_dim // 4)

        # ── Phase 2: OSM category embedding ──────────────────────────────────
        if self.use_category:
            # +1 for unknown category (id=0 reserved)
            self.category_embed = nn.Embedding(n_categories + 1, cat_dim, padding_idx=0)
            lstm_input_dim = embed_dim + embed_dim // 4 + cat_dim
        else:
            lstm_input_dim = embed_dim + embed_dim // 4
        # ─────────────────────────────────────────────────────────────────────

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc_out    = nn.Linear(hidden_dim, num_venues)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.venue_embed.weight[1:])
        nn.init.xavier_uniform_(self.time_embed.weight)
        if self.use_category:
            nn.init.xavier_uniform_(self.category_embed.weight[1:])
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(
        self,
        venue_seq:    torch.Tensor,
        time_seq:     torch.Tensor,
        category_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            venue_seq    : (batch, seq_len)  LongTensor of venue indices
            time_seq     : (batch, seq_len)  LongTensor of hour-of-day (0–23)
            category_seq : (batch, seq_len)  LongTensor of category IDs (optional)
        Returns:
            logits : (batch, num_venues)
        """
        v_emb = self.venue_embed(venue_seq)          # (B, T, embed_dim)
        t_emb = self.time_embed(time_seq)             # (B, T, embed_dim//4)

        if self.use_category and category_seq is not None:
            c_emb = self.category_embed(category_seq) # (B, T, embed_dim//4)
            x = torch.cat([v_emb, t_emb, c_emb], dim=-1)
        else:
            x = torch.cat([v_emb, t_emb], dim=-1)

        lstm_out, _ = self.lstm(x)                   # (B, T, hidden_dim)
        context     = self.attention(lstm_out)        # (B, hidden_dim)
        context     = self.dropout(context)
        logits      = self.fc_out(context)            # (B, num_venues)
        return logits

    # ── Flower FL helpers ─────────────────────────────────────────────────────

    def get_parameters(self) -> list:
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: list):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def predict_top_k(
        self,
        venue_seq:    torch.Tensor,
        time_seq:     torch.Tensor,
        category_seq: torch.Tensor = None,
        k:            int          = 10,
        exclude:      set          = None,
    ) -> list:
        self.eval()
        with torch.no_grad():
            logits = self.forward(venue_seq, time_seq, category_seq).squeeze(0)
            if exclude:
                logits[list(exclude)] = float('-inf')
            return torch.topk(logits, k).indices.tolist()


def build_model(config: dict) -> POILSTMModel:
    """
    Convenience factory.
    Config keys: num_venues, n_categories, embed_dim, hidden_dim,
                 num_layers, dropout.
    """
    return POILSTMModel(
        num_venues   = config.get('num_venues',   38333),
        n_categories = config.get('n_categories', 0),
        embed_dim    = config.get('embed_dim',    64),
        hidden_dim   = config.get('hidden_dim',   128),
        num_layers   = config.get('num_layers',   2),
        dropout      = config.get('dropout',      0.3),
    )
