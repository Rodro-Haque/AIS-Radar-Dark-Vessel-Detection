import pandas as pd

print("=" * 60)
print("DATASET 15 vs DATASET 16 COMPARISON")
print("=" * 60)

# Load Dataset 15 results
df15 = pd.read_csv('outputs_time_120/matching_results.csv')
dark15 = df15[df15['result'] == 'Potential Dark Vessel']

# Load Dataset 16 results
df16 = pd.read_csv('outputs_16/matching_results.csv')
dark16 = df16[df16['result'] == 'Potential Dark Vessel']

print(f"\n{'Metric':<30} {'Dataset 15':>12} {'Dataset 16':>12}")
print("-" * 60)
print(f"{'Total radar targets':<30} {len(df15):>12} {len(df16):>12}")
print(f"{'Matched vessels':<30} {(df15['result']=='Matched Vessel').sum():>12} {(df16['result']=='Matched Vessel').sum():>12}")
print(f"{'Dark vessel candidates':<30} {len(dark15):>12} {len(dark16):>12}")
print(f"{'Dark vessel %':<30} {100*len(dark15)/len(df15):>11.1f}% {100*len(dark16)/len(df16):>11.1f}%")

print(f"\n{'No AIS time candidates':<30} {(df15['reason']=='No AIS time candidate').sum():>12} {(df16['reason']=='No AIS time candidate').sum():>12}")
print(f"{'No AIS distance candidates':<30} {(df15['reason']=='No AIS distance candidate').sum():>12} {(df16['reason']=='No AIS distance candidate').sum():>12}")
print(f"{'Low similarity':<30} {(df15['reason']=='Similarity threshold').sum():>12} {(df16['reason']=='Similarity threshold').sum():>12}")

# Similarity statistics
print("\n" + "=" * 60)
print("SIMILARITY STATISTICS")
print("=" * 60)

for name, df in [("Dataset 15", df15), ("Dataset 16", df16)]:
    sim = df['similarity'].dropna()
    print(f"\n{name}:")
    print(f"  Mean: {sim.mean():.4f}")
    print(f"  Std:  {sim.std():.4f}")
    print(f"  Min:  {sim.min():.4f}")
    print(f"  Max:  {sim.max():.4f}")
    print(f"  10th percentile: {sim.quantile(0.10):.4f}")
    print(f"  90th percentile: {sim.quantile(0.90):.4f}")