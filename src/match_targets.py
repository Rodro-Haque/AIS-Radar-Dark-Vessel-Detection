"""
STEP 6: USE THE TRAINED MODEL (Multi-Dataset Compatible)
Test on a single held-out dataset using a model trained on multiple datasets.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch

from create_pairs import AIS_FEATURES, RADAR_FEATURES, haversine_m, ensure_timestamp_column
from model import AISRadarContrastive
from preprocess import clean_ais, clean_radar


def build_feature_row(ais_row, radar_row):
    row = {}
    for c in AIS_FEATURES:
        row["ais_" + c] = ais_row[c]
    for c in RADAR_FEATURES:
        row["radar_" + c] = radar_row[c]
    return row


def main():
    parser = argparse.ArgumentParser(description='Match radar to AIS & find dark vessels (Test on single dataset)')
    
    # Single dataset for testing
    parser.add_argument("--ais_csv", required=True, help="AIS data for matching (test set)")
    parser.add_argument("--radar_csv", required=True, help="Radar data to analyze (test set)")
    
    # Model from multi-dataset training
    parser.add_argument("--model_path", required=True, help="Trained model file")
    parser.add_argument("--scaler_path", required=True, help="Fitted scaler file")
    parser.add_argument("--out_dir", default="outputs_test", help="Output directory")
    
    # Inference settings
    parser.add_argument("--similarity_threshold", type=float, default=0.75)
    parser.add_argument("--max_time_sec", type=float, default=120)
    parser.add_argument("--max_dist_m", type=float, default=500)
    parser.add_argument("--max_radar_rows", type=int, default=None)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'='*60}")
    print("TESTING ON SINGLE DATASET")
    print(f"{'='*60}")
    print(f"AIS: {args.ais_csv}")
    print(f"Radar: {args.radar_csv}")
    print(f"Model: {args.model_path}")
    print(f"{'='*60}")

    # Load and clean test data
    print("\nLoading and cleaning test data...")
    ais_df = clean_ais(args.ais_csv)
    radar_df = clean_radar(args.radar_csv)

    ais_df = ensure_timestamp_column(ais_df)
    radar_df = ensure_timestamp_column(radar_df)

    ais_df = ais_df.dropna(subset=["timestamp", "longitude", "latitude"])
    radar_df = radar_df.dropna(subset=["timestamp", "longitude", "latitude"])

    ais_df = ais_df.sort_values("timestamp").reset_index(drop=True)
    radar_df = radar_df.sort_values("timestamp").reset_index(drop=True)

    if args.max_radar_rows is not None:
        radar_df = radar_df.head(args.max_radar_rows).copy()

    print(f"\nAIS rows: {len(ais_df)}")
    print(f"Radar rows: {len(radar_df)}")
    print(f"AIS time range: {ais_df['timestamp'].min()} to {ais_df['timestamp'].max()}")
    print(f"Radar time range: {radar_df['timestamp'].min()} to {radar_df['timestamp'].max()}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ais_dim = len(AIS_FEATURES)
    radar_dim = len(RADAR_FEATURES)

    model = AISRadarContrastive(ais_dim=ais_dim, radar_dim=radar_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Process each radar target
    print(f"\nProcessing {len(radar_df)} radar targets...")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print("-" * 50)

    results = []
    no_time_candidates = 0
    no_distance_candidates = 0

    for _, radar_row in radar_df.iterrows():
        radar_time = radar_row["timestamp"]

        t0 = radar_time - pd.Timedelta(seconds=args.max_time_sec)
        t1 = radar_time + pd.Timedelta(seconds=args.max_time_sec)

        candidates = ais_df[(ais_df["timestamp"] >= t0) & (ais_df["timestamp"] <= t1)].copy()

        if len(candidates) == 0:
            no_time_candidates += 1
            results.append({
                "targetId": radar_row.get("targetId", None),
                "radar_time": radar_time,
                "best_mmsi": None,
                "similarity": np.nan,
                "distance_m": np.nan,
                "result": "Potential Dark Vessel",
                "reason": "No AIS time candidate",
            })
            continue

        distances = haversine_m(
            radar_row["longitude"], radar_row["latitude"],
            candidates["longitude"].values, candidates["latitude"].values,
        )
        candidates["distance_m"] = distances
        candidates = candidates[candidates["distance_m"] <= args.max_dist_m].copy()

        if len(candidates) == 0:
            no_distance_candidates += 1
            results.append({
                "targetId": radar_row.get("targetId", None),
                "radar_time": radar_time,
                "best_mmsi": None,
                "similarity": np.nan,
                "distance_m": np.nan,
                "result": "Potential Dark Vessel",
                "reason": "No AIS distance candidate",
            })
            continue

        # Score candidates with model
        feature_rows = []
        for _, ais_row in candidates.iterrows():
            feature_rows.append(build_feature_row(ais_row, radar_row))

        feature_df = pd.DataFrame(feature_rows)
        feature_cols = ["ais_" + c for c in AIS_FEATURES] + ["radar_" + c for c in RADAR_FEATURES]
        X = feature_df[feature_cols].astype("float32").fillna(0).values
        Xs = scaler.transform(X)

        ais_x = torch.tensor(Xs[:, :ais_dim], dtype=torch.float32).to(device)
        radar_x = torch.tensor(Xs[:, ais_dim:], dtype=torch.float32).to(device)

        with torch.no_grad():
            ais_emb, radar_emb = model(ais_x, radar_x)
            # Euclidean distance (lower = more similar)
            dists = torch.norm(ais_emb - radar_emb, p=2, dim=1).cpu().numpy()

        best_i = int(np.argmin(dists))
        best_distance = float(dists[best_i])
        # Convert distance to similarity score (for threshold comparison)
        # Using: similarity = 1 / (1 + distance) so that lower distance = higher similarity
        best_similarity = 1.0 / (1.0 + best_distance)
        best_ais = candidates.iloc[best_i]
        best_distance = float(best_ais["distance_m"])

        if best_similarity >= args.similarity_threshold:
            result = "Matched Vessel"
        else:
            result = "Potential Dark Vessel"

        results.append({
            "targetId": radar_row.get("targetId", None),
            "radar_time": radar_time,
            "best_mmsi": best_ais.get("mmsi", None),
            "similarity": best_similarity,
            "distance_m": best_distance,
            "result": result,
            "reason": "Similarity threshold",
        })

    results_df = pd.DataFrame(results)

    # Save results
    matching_path = os.path.join(args.out_dir, "matching_results.csv")
    dark_path = os.path.join(args.out_dir, "dark_vessel_candidates.csv")

    results_df.to_csv(matching_path, index=False)
    results_df[results_df["result"] == "Potential Dark Vessel"].to_csv(dark_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results_df["result"].value_counts(dropna=False).to_string())
    print(f"\nNo AIS time candidates: {no_time_candidates}")
    print(f"No AIS distance candidates: {no_distance_candidates}")

    if results_df["similarity"].notna().sum() > 0:
        print("\nSimilarity statistics:")
        print(results_df["similarity"].describe())
        print("\nPercentiles:")
        print(results_df["similarity"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]))

    print(f"\nSaved to:")
    print(f"  - {matching_path}")
    print(f"  - {dark_path}")


if __name__ == "__main__":
    main()