"""
osm/fetcher.py
--------------
Fetch POI data from OpenStreetMap via the Overpass API.

Queries amenity, shop, and tourism nodes for a bounding box and
caches results to a local JSON file to avoid repeated API calls.

Usage:
    from osm.fetcher import fetch_osm_pois
    pois = fetch_osm_pois(bbox=(40.47, -74.26, 40.92, -73.68))
"""

import json
import time
import ssl
import urllib.request
import urllib.parse
from typing import List, Dict, Tuple

# ── SSL fix for macOS (certificate verification issue) ────────────────────────
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode    = ssl.CERT_NONE
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
OVERPASS_URL  = 'http://overpass-api.de/api/interpreter'
REQUEST_TIMEOUT = 120          # seconds
RETRY_DELAY     = 10           # seconds between retries
MAX_RETRIES     = 3

# NYC bounding box: (south, west, north, east)
NYC_BBOX = (40.4774, -74.2591, 40.9176, -73.7004)

# OSM tags to fetch — covers the main Foursquare venue categories
OSM_TAGS = ['amenity', 'shop', 'tourism', 'leisure', 'office']
# ─────────────────────────────────────────────────────────────────────────────


def _build_overpass_query(bbox: Tuple[float, float, float, float],
                          tags: List[str]) -> str:
    """
    Build an Overpass QL query that fetches nodes with any of the given tags
    within the bounding box.

    Args:
        bbox: (south, west, north, east) in decimal degrees
        tags: list of OSM tag keys to query (e.g. ['amenity', 'shop'])

    Returns:
        Overpass QL query string
    """
    south, west, north, east = bbox
    bbox_str = f'{south},{west},{north},{east}'

    node_queries = '\n'.join(
        f'  node["{tag}"]({bbox_str});'
        for tag in tags
    )

    return f"""
[out:json][timeout:120];
(
{node_queries}
);
out body;
"""


def fetch_osm_pois(
    bbox:       Tuple[float, float, float, float] = NYC_BBOX,
    tags:       List[str]                         = None,
    cache_path: str                               = 'data/osm/osm_pois_raw.json',
) -> List[Dict]:
    """
    Fetch OSM POI nodes for a bounding box via Overpass API.

    Results are cached to cache_path so subsequent calls return immediately
    without hitting the API again. Delete the cache file to force a re-fetch.

    Args:
        bbox       : (south, west, north, east) bounding box
        tags       : OSM tag keys to query. Defaults to OSM_TAGS.
        cache_path : path to JSON cache file

    Returns:
        List of POI dicts, each with keys:
            id    : OSM node ID
            lat   : latitude
            lon   : longitude
            tags  : dict of OSM tags (e.g. {'amenity': 'restaurant', 'name': '...'})
    """
    import os
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

    # ── Return from cache if available ───────────────────────────────────────
    if os.path.exists(cache_path):
        print(f'[OSM Fetcher] Loading from cache: {cache_path}')
        with open(cache_path, 'r') as f:
            data = json.load(f)
        print(f'[OSM Fetcher] {len(data):,} POIs loaded from cache.')
        return data

    if tags is None:
        tags = OSM_TAGS

    query = _build_overpass_query(bbox, tags)
    encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')

    # ── Query Overpass API with retries ───────────────────────────────────────
    print(f'[OSM Fetcher] Querying Overpass API for bbox={bbox} ...')
    print(f'[OSM Fetcher] Tags: {tags}')
    print(f'[OSM Fetcher] This may take 30–60 seconds ...')

    elements = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OVERPASS_URL,
                data    = encoded,
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent':   'FedDP-POI/1.0 (thesis research)',
                    'Accept':       '*/*',
                },
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            elements = result.get('elements', [])
            break
        except Exception as e:
            print(f'[OSM Fetcher] Attempt {attempt}/{MAX_RETRIES} failed: {e}')
            if attempt < MAX_RETRIES:
                print(f'[OSM Fetcher] Retrying in {RETRY_DELAY}s ...')
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    'Overpass API request failed after all retries. '
                    'Check your internet connection or try again later.'
                ) from e

    # ── Parse nodes ───────────────────────────────────────────────────────────
    pois = []
    for el in elements:
        if el.get('type') != 'node':
            continue
        if 'lat' not in el or 'lon' not in el:
            continue
        pois.append({
            'id':   el['id'],
            'lat':  el['lat'],
            'lon':  el['lon'],
            'tags': el.get('tags', {}),
        })

    print(f'[OSM Fetcher] Fetched {len(pois):,} POI nodes.')

    # ── Save to cache ─────────────────────────────────────────────────────────
    with open(cache_path, 'w') as f:
        json.dump(pois, f)
    print(f'[OSM Fetcher] Cached to {cache_path}')

    return pois


def get_osm_category(tags: Dict[str, str]) -> str:
    """
    Extract a single category label from an OSM node's tags.
    Priority: amenity > shop > tourism > leisure > office > 'unknown'

    Args:
        tags: dict of OSM tags for a node

    Returns:
        Category string e.g. 'restaurant', 'supermarket', 'hotel', 'unknown'
    """
    for key in OSM_TAGS:
        if key in tags:
            return tags[key]
    return 'unknown'