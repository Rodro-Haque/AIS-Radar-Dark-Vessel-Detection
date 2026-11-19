import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import math


AIS_FEATURES = ['latitude', 'longitude', 'sog', 'cog']
RADAR_FEATURES = ['latitude', 'longitude', 'sog', 'cog', 'signalLevel']


def haversine_m(lon1, lat1, lon2, lat2):
    """
    Calculate haversine distance. Works with both scalars and numpy arrays.
    """
    R = 6371000
    # Convert to numpy arrays for vectorized operations
    lat1 = np.asarray(lat1)
    lat2 = np.asarray(lat2)
    lon1 = np.asarray(lon1)
    lon2 = np.asarray(lon2)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def ensure_timestamp_column(df):
    if 'timestamp' not in df.columns:
        for col in ['date_time', 'dateTime', 'datetime']:
            if col in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                break
    return df


def create_pairs_single_dataset(ais_df, radar_df, max_time_sec=120, max_dist_m=500,
                                 n_pos_target=3000, n_neg_target=9000, random_state=42):
    """
    Generate controlled number of positive and negative pairs from a SINGLE dataset.
    Uses HARD NEGATIVE MINING: negatives are close in time/space but different vessels.

    Args:
        ais_df: AIS DataFrame (single dataset)
        radar_df: Radar DataFrame (single dataset)
        max_time_sec: Maximum time difference for positive pairs
        max_dist_m: Maximum distance for positive pairs
        n_pos_target: Target number of positive pairs
        n_neg_target: Target number of negative pairs
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with pairs
    """
    rng = np.random.RandomState(random_state)

    ais_df = ensure_timestamp_column(ais_df)
    radar_df = ensure_timestamp_column(radar_df)

    ais_df = ais_df.dropna(subset=['timestamp', 'longitude', 'latitude']).copy()
    radar_df = radar_df.dropna(subset=['timestamp', 'longitude', 'latitude']).copy()

    ais_df = ais_df.sort_values('timestamp').reset_index(drop=True)
    radar_df = radar_df.sort_values('timestamp').reset_index(drop=True)

    # Build BallTree on radar for fast spatial lookup
    radar_coords = np.radians(radar_df[['latitude', 'longitude']].values)
    tree = BallTree(radar_coords, metric='haversine')
    max_dist_rad = max_dist_m / 6371000.0

    # ==================== POSITIVE PAIRS ====================
    positive_pairs = []

    for idx_ais, row_ais in ais_df.iterrows():
        t_ais = row_ais['timestamp']
        lat_ais = row_ais['latitude']
        lon_ais = row_ais['longitude']

        query_coord = np.radians([[lat_ais, lon_ais]])
        indices = tree.query_radius(query_coord, r=max_dist_rad)[0]

        for idx_rad in indices:
            row_rad = radar_df.iloc[idx_rad]
            t_rad = row_rad['timestamp']
            time_diff = abs((t_ais - t_rad).total_seconds())

            if time_diff <= max_time_sec:
                positive_pairs.append((row_ais, row_rad, 1))

    # Limit positive pairs to target
    if len(positive_pairs) > n_pos_target:
        indices = rng.choice(len(positive_pairs), size=n_pos_target, replace=False)
        positive_pairs = [positive_pairs[i] for i in indices]

    n_pos = len(positive_pairs)
    print(f"  Positive pairs generated: {n_pos} (target: {n_pos_target})")

    # ==================== HARD NEGATIVE PAIRS ====================
    # Hard negatives: close in time/space but from DIFFERENT vessels
    # We approximate this by finding AIS-Radar pairs within window
    # but sampling from mismatched pairs

    hard_negative_pairs = []

    # Strategy: For each radar target, find AIS candidates in window
    # Then pick AIS that is NOT the best match (different vessel)
    for idx_rad, row_rad in radar_df.sample(min(len(radar_df), n_neg_target * 2), 
                                             random_state=rng.randint(0, 2**31)).iterrows():
        if len(hard_negative_pairs) >= n_neg_target:
            break

        t_rad = row_rad['timestamp']
        lat_rad = row_rad['latitude']
        lon_rad = row_rad['longitude']

        # Find AIS in time window
        t0 = t_rad - pd.Timedelta(seconds=max_time_sec)
        t1 = t_rad + pd.Timedelta(seconds=max_time_sec)
        time_candidates = ais_df[(ais_df['timestamp'] >= t0) & (ais_df['timestamp'] <= t1)].copy()

        if len(time_candidates) == 0:
            continue

        # Find AIS in distance window
        distances = haversine_m(
            lon_rad, lat_rad,
            time_candidates['longitude'].values, time_candidates['latitude'].values,
        )
        time_candidates['distance_m'] = distances
        dist_candidates = time_candidates[time_candidates['distance_m'] <= max_dist_m]

        if len(dist_candidates) < 2:
            # Not enough candidates for a hard negative, use random
            row_ais = ais_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
            hard_negative_pairs.append((row_ais, row_rad, -1))
        else:
            # Pick a random candidate that is NOT the closest one
            # (Closest = likely same vessel, others = hard negatives)
            dist_candidates = dist_candidates.sort_values('distance_m')
            # Exclude the closest match
            hard_candidates = dist_candidates.iloc[1:]  # Skip closest
            if len(hard_candidates) > 0:
                row_ais = hard_candidates.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
                hard_negative_pairs.append((row_ais, row_rad, -1))

    # Fill remaining with random negatives if needed
    n_hard = len(hard_negative_pairs)
    n_random = max(0, n_neg_target - n_hard)

    for _ in range(n_random):
        row_ais = ais_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
        row_rad = radar_df.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
        hard_negative_pairs.append((row_ais, row_rad, -1))

    n_neg = len(hard_negative_pairs)
    print(f"  Hard negative pairs: {n_hard}")
    print(f"  Random negative pairs: {n_random}")
    print(f"  Total negative pairs: {n_neg} (target: {n_neg_target})")

    # Combine and build DataFrame
    all_pairs = positive_pairs + hard_negative_pairs

    pair_data = []
    for row_ais, row_rad, label in all_pairs:
        features = {}
        for c in AIS_FEATURES:
            features['ais_' + c] = row_ais.get(c, np.nan)
        for c in RADAR_FEATURES:
            features['radar_' + c] = row_rad.get(c, np.nan)
        features['label'] = label
        pair_data.append(features)

    pairs_df = pd.DataFrame(pair_data)
    return pairs_df

def create_pairs(ais_df, radar_df, max_time_sec=120, max_dist_m=500, max_pairs=50000):
    """
    Backward-compatible wrapper that works with combined DataFrames.
    If dataset_id column exists, generates controlled pairs per dataset.
    Otherwise, falls back to old behavior.
    """
    # Check if we have multiple datasets
    if 'dataset_id' in ais_df.columns and 'dataset_id' in radar_df.columns:
        all_pairs = []
        dataset_ids = sorted(ais_df['dataset_id'].unique())

        for ds_id in dataset_ids:
            print(f"\n--- Processing dataset_id={ds_id} ---")
            ds_ais = ais_df[ais_df['dataset_id'] == ds_id].copy()
            ds_radar = radar_df[radar_df['dataset_id'] == ds_id].copy()

            ds_pairs = create_pairs_single_dataset(
                ds_ais, ds_radar,
                max_time_sec=max_time_sec,
                max_dist_m=max_dist_m,
                n_pos_target=3000,
                n_neg_target=9000,
                random_state=42 + ds_id
            )
            ds_pairs['dataset_id'] = ds_id
            all_pairs.append(ds_pairs)

        combined = pd.concat(all_pairs, ignore_index=True)
        print(f"\n=== TOTAL PAIRS ===")
        print(f"  Total: {len(combined)}")
        print(f"  Label distribution:\n{combined['label'].value_counts().to_string()}")
        return combined
    else:
        # Single dataset — use old behavior
        return create_pairs_single_dataset(
            ais_df, radar_df,
            max_time_sec=max_time_sec,
            max_dist_m=max_dist_m,
            n_pos_target=3000,
            n_neg_target=9000
        )