import sys
import os
# Add the src folder to Python's path so it can find preprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from preprocess import clean_ais, clean_radar

print("=" * 60)
print("TIMING INVESTIGATION")
print("=" * 60)

# 1. Check original radar data
print("\n--- 1. ORIGINAL RADAR CSV ---")
radar_raw = pd.read_csv("data/Radar-15.csv")
print(f"Total rows: {len(radar_raw)}")
print(f"dateTime column sample:")
print(radar_raw['dateTime'].head(3).tolist())
print(f"dateTime column tail:")
print(radar_raw['dateTime'].tail(3).tolist())

# 2. Check preprocessed radar (what match_targets uses)
print("\n--- 2. PREPROCESSED RADAR (after clean_radar) ---")
from preprocess import clean_radar
radar_clean = clean_radar("data/Radar-15.csv")
print(f"Total rows after cleaning: {len(radar_clean)}")
print(f"Time range: {radar_clean['dateTime'].min()} to {radar_clean['dateTime'].max()}")

# 3. Check matching results
print("\n--- 3. MATCHING RESULTS ---")
results = pd.read_csv("outputs_threshold_75/matching_results.csv")
results['radar_time'] = pd.to_datetime(results['radar_time'])
print(f"Total results: {len(results)}")
print(f"Time range: {results['radar_time'].min()} to {results['radar_time'].max()}")

# 4. Compare row counts
print(f"\n--- 4. ROW COUNT CHECK ---")
print(f"Original radar rows: {len(radar_raw)}")
print(f"Cleaned radar rows:  {len(radar_clean)}")
print(f"Results rows:        {len(results)}")
if len(radar_clean) == len(results):
    print("✓ Counts match — every radar row has one result")
else:
    print(f"✗ MISMATCH! Difference: {len(results) - len(radar_clean)}")

# 5. Dark vessel time distribution
print("\n--- 5. DARK VESSEL TIMES ---")
dark = results[results['result'] == 'Potential Dark Vessel']
print(f"Dark vessels: {len(dark)}")
print(f"Dark time range: {dark['radar_time'].min()} to {dark['radar_time'].max()}")

# Show dark vessels by 5-minute bins
dark['time_bin'] = dark['radar_time'].dt.floor('5min')
print("\nDark vessels by 5-minute window:")
print(dark['time_bin'].value_counts().sort_index())

# 6. Check if "No AIS distance" cases cluster at end
print("\n--- 6. NO-AIS CASES BY TIME ---")
no_ais = dark[dark['reason'] == 'No AIS distance candidate']
if len(no_ais) > 0:
    no_ais['time_bin'] = no_ais['radar_time'].dt.floor('5min')
    print(f"No-AIS dark vessels: {len(no_ais)}")
    print(no_ais['time_bin'].value_counts().sort_index().tail(10))