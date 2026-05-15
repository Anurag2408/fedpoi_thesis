"""
osm/matcher.py
--------------
Match Foursquare venues to the nearest OSM POI using Haversine distance.

For each Foursquare venue (latitude, longitude), we find the closest OSM node
within a configurable radius and extract its amenity category. Venues with no
OSM match within the radius are assigned category 'unknown' (id=0).

Usage:
    from osm.matcher import match_venues_to_osm, build_category_mapping
    matches      = match_venues_to_osm(venues_df, osm_pois)
    cat_mapping  = build_category_mapping(matches)
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from .fetcher import get_osm_category

# ── Config ────────────────────────────────────────────────────────────────────
MAX_MATCH_DIST_M = 200      # maximum metres to consider a valid match
EARTH_RADIUS_M   = 6_371_000
# ─────────────────────────────────────────────────────────────────────────────

# Broad OSM category → thesis-level super-category mapping
# Reduces the 300+ OSM amenity types to ~15 meaningful groups
SUPER_CATEGORY_MAP = {
    # Food & Drink
    'restaurant': 'food', 'cafe': 'food', 'bar': 'food', 'pub': 'food',
    'fast_food': 'food', 'food_court': 'food', 'ice_cream': 'food',
    'bakery': 'food', 'biergarten': 'food', 'food': 'food',
    # Shopping
    'supermarket': 'shop', 'convenience': 'shop', 'clothes': 'shop',
    'mall': 'shop', 'department_store': 'shop', 'electronics': 'shop',
    'shop': 'shop', 'marketplace': 'shop', 'gift': 'shop', 'books': 'shop',
    # Arts & Entertainment
    'cinema': 'entertainment', 'theatre': 'entertainment',
    'nightclub': 'entertainment', 'casino': 'entertainment',
    'arts_centre': 'entertainment', 'museum': 'entertainment',
    'gallery': 'entertainment', 'attraction': 'entertainment',
    # Travel & Transport
    'hotel': 'travel', 'hostel': 'travel', 'motel': 'travel',
    'guest_house': 'travel', 'bus_station': 'travel', 'taxi': 'travel',
    'car_rental': 'travel', 'parking': 'travel',
    # Outdoors & Recreation
    'park': 'outdoors', 'playground': 'outdoors', 'sports_centre': 'outdoors',
    'gym': 'outdoors', 'fitness_centre': 'outdoors', 'stadium': 'outdoors',
    'leisure': 'outdoors', 'swimming_pool': 'outdoors',
    # Education
    'school': 'education', 'university': 'education', 'college': 'education',
    'library': 'education', 'kindergarten': 'education',
    # Medical
    'hospital': 'medical', 'clinic': 'medical', 'pharmacy': 'medical',
    'doctors': 'medical', 'dentist': 'medical',
    # Financial
    'bank': 'financial', 'atm': 'financial', 'bureau_de_change': 'financial',
    # Government & Services
    'post_office': 'services', 'police': 'services', 'fire_station': 'services',
    'townhall': 'services', 'courthouse': 'services', 'embassy': 'services',
    # Religious
    'place_of_worship': 'religious', 'church': 'religious',
    'mosque': 'religious', 'temple': 'religious',
    # Office
    'office': 'office',
}

SUPER_CATEGORIES = [
    'unknown', 'food', 'shop', 'entertainment', 'travel',
    'outdoors', 'education', 'medical', 'financial',
    'services', 'religious', 'office', 'other',
]
N_SUPER_CATEGORIES = len(SUPER_CATEGORIES)   # 13


def haversine_m(lat1: float, lon1: float,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorised Haversine distance from a single point to an array of points.

    Args:
        lat1, lon1 : reference point (degrees)
        lat2, lon2 : arrays of candidate points (degrees)

    Returns:
        Array of distances in metres
    """
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def osm_to_super(raw_category: str) -> str:
    """Map a raw OSM category string to a super-category."""
    cat = SUPER_CATEGORY_MAP.get(raw_category)
    if cat:
        return cat
    # Partial matches (e.g. 'fast_food' → 'food')
    for key, val in SUPER_CATEGORY_MAP.items():
        if key in raw_category or raw_category in key:
            return val
    return 'other'


def match_venues_to_osm(
    venues_df:    pd.DataFrame,
    osm_pois:     List[Dict],
    max_dist_m:   float = MAX_MATCH_DIST_M,
    id_col:       str   = 'venue_id_encoded',
    lat_col:      str   = 'latitude',
    lon_col:      str   = 'longitude',
) -> Dict[int, Dict]:
    """
    Match each Foursquare venue to the nearest OSM POI within max_dist_m.

    Args:
        venues_df  : DataFrame with venue_id_encoded, latitude, longitude columns
                     (use the deduplicated venue list from foursquare_filtered.csv)
        osm_pois   : list of OSM POI dicts from fetcher.fetch_osm_pois()
        max_dist_m : maximum match radius in metres
        id_col     : column name for venue encoded ID
        lat_col    : column name for venue latitude
        lon_col    : column name for venue longitude

    Returns:
        Dict mapping venue_id_encoded → {
            'raw_category'   : str   (OSM amenity type, e.g. 'restaurant')
            'super_category' : str   (grouped label, e.g. 'food')
            'distance_m'     : float (distance to matched OSM node)
            'matched'        : bool  (False if no OSM node within max_dist_m)
        }
    """
    if not osm_pois:
        raise ValueError('osm_pois list is empty — run fetch_osm_pois() first.')

    # Build OSM arrays for fast vectorised distance computation
    osm_lats = np.array([p['lat'] for p in osm_pois])
    osm_lons = np.array([p['lon'] for p in osm_pois])
    osm_cats = [get_osm_category(p['tags']) for p in osm_pois]

    # Deduplicate venues (one row per unique venue)
    venue_df = venues_df[[id_col, lat_col, lon_col]].drop_duplicates(
        subset=id_col
    ).dropna(subset=[lat_col, lon_col])

    results  = {}
    n_matched = 0

    print(f'[OSM Matcher] Matching {len(venue_df):,} venues to '
          f'{len(osm_pois):,} OSM nodes (max_dist={max_dist_m}m) ...')

    for _, row in venue_df.iterrows():
        vid  = int(row[id_col])
        vlat = float(row[lat_col])
        vlon = float(row[lon_col])

        dists    = haversine_m(vlat, vlon, osm_lats, osm_lons)
        min_idx  = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= max_dist_m:
            raw_cat   = osm_cats[min_idx]
            super_cat = osm_to_super(raw_cat)
            matched   = True
            n_matched += 1
        else:
            raw_cat   = 'unknown'
            super_cat = 'unknown'
            matched   = False

        results[vid] = {
            'raw_category':   raw_cat,
            'super_category': super_cat,
            'distance_m':     min_dist,
            'matched':        matched,
        }

    match_rate = n_matched / max(len(venue_df), 1) * 100
    print(f'[OSM Matcher] Matched {n_matched:,}/{len(venue_df):,} venues '
          f'({match_rate:.1f}% match rate)')

    return results


def build_category_mapping(
    matches: Dict[int, Dict],
) -> Tuple[Dict[int, int], Dict[str, int], int]:
    """
    Build integer encoding for venue categories.

    Args:
        matches : output of match_venues_to_osm()

    Returns:
        venue_categories : Dict[venue_id_encoded → category_id (int)]
        category_mapping : Dict[category_name → category_id (int)]
        n_categories     : total number of unique categories (including 'unknown')
    """
    # Build category → id mapping using fixed SUPER_CATEGORIES order
    # so IDs are stable across runs
    category_mapping = {cat: idx for idx, cat in enumerate(SUPER_CATEGORIES)}

    venue_categories = {}
    for vid, info in matches.items():
        super_cat = info['super_category']
        cat_id    = category_mapping.get(super_cat,
                    category_mapping['other'])
        venue_categories[vid] = cat_id

    n_categories = len(SUPER_CATEGORIES)

    # Print distribution
    from collections import Counter
    dist = Counter(info['super_category'] for info in matches.values())
    print('\n[OSM Matcher] Category distribution:')
    for cat, count in dist.most_common():
        bar = '█' * int(count / max(dist.values()) * 30)
        print(f'  {cat:<14}: {count:>6,}  {bar}')

    return venue_categories, category_mapping, n_categories
