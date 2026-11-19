import pandas as pd
import os

configs = [
    ('outputs_15_multi/matching_results.csv', 'Train:15+16, Test:15'),
    ('outputs_16_multi/matching_results.csv', 'Train:15+16, Test:16'),
    ('outputs_17_multi/matching_results.csv', 'Train:15+16, Test:17'),
]

print('=' * 65)
print('MULTI-DATASET TRAINING RESULTS')
print('=' * 65)
print(f"{'Config':<35} {'Total':>8} {'Matched':>8} {'Dark':>8} {'Match%':>8}")
print('-' * 65)

for path, name in configs:
    if os.path.exists(path):
        df = pd.read_csv(path)
        total = len(df)
        matched = (df['result'] == 'Matched Vessel').sum()
        dark = (df['result'] == 'Potential Dark Vessel').sum()
        match_pct = 100 * matched / total
        print(f"{name:<35} {total:>8} {matched:>8} {dark:>8} {match_pct:>7.1f}%")
    else:
        print(f"{name:<35} {'NOT RUN':>30}")

print('=' * 65)