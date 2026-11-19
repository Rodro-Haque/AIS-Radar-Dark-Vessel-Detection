import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("outputs_threshold_75/matching_results.csv")
dark = df[df["result"] == "Potential Dark Vessel"].copy()

print("=" * 60)
print("DARK VESSEL ANALYSIS")
print("=" * 60)

print(f"\nTotal dark vessel candidates: {len(dark)}")

# Break down by reason
print("\n--- By Reason ---")
print(dark["reason"].value_counts())

# For cases with similarity scores, show distribution
has_sim = dark["similarity"].notna()
print(f"\n--- Cases with similarity scores: {has_sim.sum()} ---")
print(f"Cases without similarity (no AIS nearby): {(~has_sim).sum()}")

if has_sim.sum() > 0:
    sim_values = dark.loc[has_sim, "similarity"]
    print(f"\nSimilarity distribution for dark vessels WITH AIS candidates:")
    print(f"  Min: {sim_values.min():.3f}")
    print(f"  Max: {sim_values.max():.3f}")
    print(f"  Mean: {sim_values.mean():.3f}")
    print(f"  Median: {sim_values.median():.3f}")
    
    print(f"\n  Binned distribution:")
    bins = [0, 0.5, 0.6, 0.7, 0.75]
    for i in range(len(bins)-1):
        count = ((sim_values >= bins[i]) & (sim_values < bins[i+1])).sum()
        print(f"    {bins[i]:.2f} - {bins[i+1]:.2f}: {count}")

# Look at distance for cases with AIS candidates
has_dist = dark["distance_m"].notna()
if has_dist.sum() > 0:
    dist_values = dark.loc[has_dist, "distance_m"]
    print(f"\n--- Distance distribution (for cases with AIS nearby) ---")
    print(f"  Min: {dist_values.min():.1f} m")
    print(f"  Max: {dist_values.max():.1f} m")
    print(f"  Mean: {dist_values.mean():.1f} m")
    print(f"  Median: {dist_values.median():.1f} m")
    
    print(f"\n  Distance bins:")
    for th in [100, 200, 300, 400, 500]:
        count = (dist_values <= th).sum()
        print(f"    ≤ {th} m: {count}")

# Check if dark vessels cluster in time
print("\n--- Temporal patterns ---")
dark["radar_time"] = pd.to_datetime(dark["radar_time"])
print("Dark vessels by minute:")
print(dark["radar_time"].dt.floor("1min").value_counts().head(10))

# Check if dark vessels cluster in space
print("\n--- Spatial patterns ---")
# Get actual column names
print("Available columns:", dark.columns.tolist())

# Try common column names for coordinates
lat_col = None
lon_col = None
for col in dark.columns:
    if 'lat' in col.lower():
        lat_col = col
    if 'lon' in col.lower():
        lon_col = col

if lat_col and lon_col:
    print(f"Latitude range: {dark[lat_col].min():.4f} to {dark[lat_col].max():.4f}")
    print(f"Longitude range: {dark[lon_col].min():.4f} to {dark[lon_col].max():.4f}")
else:
    print("Could not find lat/lon columns automatically")