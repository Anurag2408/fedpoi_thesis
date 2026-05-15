# FedDP-POI: Federated POI Recommendation with Adaptive Differential Privacy

M.Tech thesis project — privacy-preserving federated learning for point-of-interest recommendation using LSTM + OpenStreetMap features + Adaptive DP-SGD.

## Repository Structure

```
fedpoi_thesis/
├── .gitignore
├── README.md
├── requirements.txt
├── evaluate.py                   # Shared evaluation utilities (P@K, R@K, NDCG@K, F1@K)
├── prepare_sequential_data.py    # Foursquare → federated client splits
├── model/
│   ├── __init__.py
│   ├── lstm_model.py             # LSTM with venue + time + category embeddings
│   └── dataset.py                # POISequenceDataset and get_dataloaders
├── osm/
│   ├── __init__.py
│   ├── fetcher.py                # Overpass API POI fetcher
│   ├── matcher.py                # Haversine venue-to-POI matcher
│   └── prepare_osm_features.py   # Pipeline orchestrator
├── privacy/
│   ├── __init__.py
│   ├── budget_allocator.py       # Adaptive ε per client (linear interpolation)
│   ├── dp_engine.py              # DP-SGD: clip → Gaussian noise → step
│   └── accountant.py             # Per-client privacy budget tracker
├── experiments/
│   ├── phase3_adaptive_dp/
│   │   ├── client.py             # Full system: LSTM + OSM + Adaptive DP-SGD
│   │   └── server.py             # FedAvg + epsilon aggregation + checkpointing
│   ├── ablation1_nodp/
│   │   ├── client.py             # No DP: standard SGD with gradient clipping
│   │   └── server.py
│   ├── ablation2_fixedeps/
│   │   ├── client.py             # Fixed ε=1.0 for all clients (no adaptive alloc)
│   │   └── server.py
│   └── ablation3_noosm/
│       ├── client.py             # No OSM: n_categories=0, venue_categories={}
│       └── server.py
├── results/
│   ├── phase3_adaptive_dp/       # metrics.json + privacy_log_client_*.json
│   ├── phase2_osm/               # metrics_phase2.json
│   ├── ablation1_nodp/           # generated on first run
│   ├── ablation2_fixedeps/       # metrics.json + privacy_log_client_*.json
│   └── ablation3_noosm/          # generated on first run
└── data/
    ├── raw/                      # Foursquare NYC/TKY CSV (download separately)
    ├── processed/                # Encoder PKLs, interaction matrix
    ├── federated/                # Per-client splits (client_0..9/)
    └── osm/                      # venue_categories.pkl, category_mapping.pkl
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

Each experiment is a pair of server + client scripts. Start the server first, then launch one client per terminal (or use `run_federated.py`).

### Phase 3 — Adaptive DP (main system)

```bash
python experiments/phase3_adaptive_dp/server.py --rounds 10 --clients 10
python experiments/phase3_adaptive_dp/client.py 0   # in separate terminals
```

### Ablation 1 — No Differential Privacy

```bash
python experiments/ablation1_nodp/server.py --rounds 10 --clients 10
python experiments/ablation1_nodp/client.py 0
```

### Ablation 2 — Fixed ε = 1.0

```bash
python experiments/ablation2_fixedeps/server.py --rounds 10 --clients 10
python experiments/ablation2_fixedeps/client.py 0
```

### Ablation 3 — No OSM Features

```bash
python experiments/ablation3_noosm/server.py --rounds 10 --clients 10
python experiments/ablation3_noosm/client.py 0
```

## Data Preparation

```bash
# 1. Prepare federated client splits
python prepare_sequential_data.py

# 2. Fetch and match OSM POI features
python osm/prepare_osm_features.py
```

## Legacy Directories

`fedpoi_phase1/` and `fedpoi_ablation1_nodp/` contain the original flat experiment scripts and are kept for reference. They are excluded from git tracking via `.gitignore`.
