import pandas as pd
import numpy as np

# Load 120s/0.75 results (balanced)
df = pd.read_csv('outputs_time_120/matching_results.csv')

# Add priority column
def assign_priority(row):
    if row['result'] == 'Matched Vessel':
        return 'Matched'
    elif row['reason'] == 'No AIS distance candidate':
        return 'P1-Critical'
    elif row['similarity'] >= 0.65:
        return 'P3-Medium'
    else:
        return 'P4-Low'

df['priority'] = df.apply(assign_priority, axis=1)

# Save full report
df.to_csv('outputs_time_120/priority_report.csv', index=False)

# Print summary
print("=" * 60)
print("DARK VESSEL PRIORITY REPORT")
print("=" * 60)
print(df['priority'].value_counts().to_string())

# Print P1 details
p1 = df[df['priority'] == 'P1-Critical']
print(f"\nP1-Critical (No AIS within 500m): {len(p1)}")
print(f"  Time range: {p1['radar_time'].min()} to {p1['radar_time'].max()}")
print(f"  Median distance: {p1['distance_m'].median():.1f} m (NaN = no candidate)")

# Print P3 details
p3 = df[df['priority'] == 'P3-Medium']
if len(p3) > 0:
    print(f"\nP3-Medium (Similarity 0.65-0.75): {len(p3)}")
    print(f"  Similarity range: {p3['similarity'].min():.3f} to {p3['similarity'].max():.3f}")
    print(f"  Distance range: {p3['distance_m'].min():.1f} to {p3['distance_m'].max():.1f} m")

print(f"\nSaved full report to: outputs_time_120/priority_report.csv")