"""
osm/prepare_osm_features.py
----------------------------
One-time pipeline: fetch OSM POIs → match to Foursquare venues → save features.

Run this ONCE before Phase 2 federated training.

Usage:
    python osm/prepare_osm_features.py

Outputs:
    data/osm/osm_pois_raw.json          ← raw Overpass API response (cache)
    data/osm/venue_categories.pkl       ← {venue_id_encoded: category_id}
    data/osm/category_mapping.pkl       ← {category_name: category_id}
    data/osm/match_stats.json           ← match statistics for thesis reporting
"""

import os
import sys
import json
import pickle
import pandas as pd

# Ensure project root is in path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm.fetcher import fetch_osm_pois, NYC_BBOX
from osm.matcher import match_venues_to_osm, build_category_mapping, N_SUPER_CATEGORIES

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_CSV    = 'data/processed/foursquare_filtered.csv'
OSM_DIR          = 'data/osm'
RAW_CACHE        = os.path.join(OSM_DIR, 'osm_pois_raw.json')
VENUE_CATS_PATH  = os.path.join(OSM_DIR, 'venue_categories.pkl')
CAT_MAP_PATH     = os.path.join(OSM_DIR, 'category_mapping.pkl')
STATS_PATH       = os.path.join(OSM_DIR, 'match_stats.json')
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OSM_DIR, exist_ok=True)

    print('=' * 60)
    print('  OSM FEATURE PREPARATION — FedDP-POI Phase 2')
    print('=' * 60)

    # ── Step 1: Load processed venue data ────────────────────────────────────
    print(f'\n[1/4] Loading venues from {PROCESSED_CSV} ...')
    df = pd.read_csv(PROCESSED_CSV)

    # Keep only the columns we need, deduplicated by venue
    venues_df = df[['venue_id_encoded', 'latitude', 'longitude']].drop_duplicates(
        subset='venue_id_encoded'
    )
    print(f'      {len(venues_df):,} unique venues loaded.')

    # Sanity check — ensure lat/lon exist
    missing = venues_df[['latitude', 'longitude']].isnull().sum().sum()
    if missing > 0:
        print(f'      WARNING: {missing} missing lat/lon values — '
              f'those venues will be assigned category "unknown".')

    # ── Step 2: Fetch OSM POIs ────────────────────────────────────────────────
    print(f'\n[2/4] Fetching OSM POIs for NYC ...')
    osm_pois = fetch_osm_pois(bbox=NYC_BBOX, cache_path=RAW_CACHE)
    print(f'      {len(osm_pois):,} OSM nodes available for matching.')

    # ── Step 3: Match venues to OSM ───────────────────────────────────────────
    print(f'\n[3/4] Matching Foursquare venues to nearest OSM POI ...')
    matches = match_venues_to_osm(
        venues_df  = venues_df,
        osm_pois   = osm_pois,
        max_dist_m = 200,
    )

    venue_categories, category_mapping, n_categories = build_category_mapping(matches)

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    print(f'\n[4/4] Saving feature files ...')

    with open(VENUE_CATS_PATH, 'wb') as f:
        pickle.dump(venue_categories, f)
    print(f'      venue_categories.pkl → {len(venue_categories):,} venues')

    with open(CAT_MAP_PATH, 'wb') as f:
        pickle.dump(category_mapping, f)
    print(f'      category_mapping.pkl → {n_categories} categories')

    # Save stats for thesis reporting
    n_matched  = sum(1 for m in matches.values() if m['matched'])
    n_total    = len(matches)
    stats = {
        'n_venues':          n_total,
        'n_osm_pois':        len(osm_pois),
        'n_matched':         n_matched,
        'match_rate_pct':    round(n_matched / max(n_total, 1) * 100, 2),
        'n_categories':      n_categories,
        'category_mapping':  category_mapping,
        'category_counts':   {},
    }
    from collections import Counter
    dist = Counter(m['super_category'] for m in matches.values())
    stats['category_counts'] = dict(dist.most_common())

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'      match_stats.json saved.')

    print(f'\n{"="*60}')
    print(f'  OSM preparation complete.')
    print(f'  Match rate   : {stats["match_rate_pct"]:.1f}%')
    print(f'  Categories   : {n_categories}')
    print(f'  Ready for Phase 2 federated training.')
    print(f'{"="*60}\n')

    # Print category mapping for reference
    print('  Category ID mapping (use in MODEL_CONFIG):')
    for name, cid in sorted(category_mapping.items(), key=lambda x: x[1]):
        print(f'    {cid:>2}: {name}')
    print(f'\n  Set in 06_federated_client.py:')
    print(f"    MODEL_CONFIG['n_categories'] = {n_categories}")


if __name__ == '__main__':
    main()
