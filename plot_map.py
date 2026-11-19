import sys
import os

# Add src folder to Python path so we can import preprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import matplotlib.pyplot as plt
from preprocess import clean_radar

print("Loading data...")

# Load results
results = pd.read_csv('outputs_time_120/matching_results.csv')

# Load CLEANED radar data (same as match_targets uses)
radar_clean = clean_radar("data/Radar-15.csv")

print(f"Results rows: {len(results)}")
print(f"Radar rows: {len(radar_clean)}")

# Check if row counts match
if len(results) != len(radar_clean):
    print(f"WARNING: Row count mismatch! Results={len(results)}, Radar={len(radar_clean)}")
    # Use only the first N rows that match
    n = min(len(results), len(radar_clean))
    results = results.head(n)
    radar_clean = radar_clean.head(n)
    print(f"Using first {n} rows for alignment")

# Create priority
def get_priority(row):
    if row['result'] == 'Matched Vessel':
        return 'Matched'
    elif row['reason'] == 'No AIS distance candidate':
        return 'P1-Critical'
    elif pd.notna(row['similarity']) and row['similarity'] >= 0.65:
        return 'P3-Medium'
    else:
        return 'P4-Low'

results['priority'] = results.apply(get_priority, axis=1)

# Add coordinates directly (row-by-row alignment)
results['longitude'] = radar_clean['longitude'].values
results['latitude'] = radar_clean['latitude'].values

# Drop any NaN coordinates
results = results.dropna(subset=['longitude', 'latitude'])

print(f"\nTotal rows with coordinates: {len(results)}")
for p in ['Matched', 'P1-Critical', 'P3-Medium', 'P4-Low']:
    count = (results['priority'] == p).sum()
    print(f"  {p}: {count}")

print(f"\nCoordinate ranges:")
print(f"  Longitude: {results['longitude'].min():.3f} to {results['longitude'].max():.3f}")
print(f"  Latitude: {results['latitude'].min():.3f} to {results['latitude'].max():.3f}")

# Create plot
fig, ax = plt.subplots(figsize=(12, 10))

# All matched points (tiny gray dots)
matched = results[results['priority'] == 'Matched']
ax.scatter(matched['longitude'], matched['latitude'], 
           c='lightgray', s=1, alpha=0.3, label=f'Matched ({len(matched)})')

# Dark vessels by priority
colors = {'P1-Critical': 'red', 'P3-Medium': 'orange', 'P4-Low': 'gold'}
for p, c in colors.items():
    s = results[results['priority'] == p]
    if len(s) > 0:
        ax.scatter(s['longitude'], s['latitude'], c=c, s=30, 
                  alpha=0.9, edgecolors='black', linewidth=0.5, label=f'{p} ({len(s)})')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Dark Vessel Candidates by Priority\n(120s window, 0.75 threshold)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('dark_vessel_map_v5.png', dpi=150)
plt.show()
print("Done! Saved to dark_vessel_map_v5.png")