"""
BONUS: DEBUGGING TOOL
---------------------
Use this BEFORE training to check if your data makes sense.
It shows:
- How many radar targets have nearby AIS in time/space
- Distance distribution
- Whether your thresholds are reasonable
"""

import argparse
import numpy as np
import pandas as pd
from preprocess import clean_ais, clean_radar, haversine_m


def main():
    ap = argparse.ArgumentParser(description='Debug AIS-Radar pairing')
    ap.add_argument('--ais_csv', required=True)
    ap.add_argument('--radar_csv', required=True)
    ap.add_argument('--max_time_sec', type=int, default=60, help='Time window to test')
    ap.add_argument('--max_radar_rows', type=int, default=3000, help='Limit for speed')
    args = ap.parse_args()
    
    # Load data
    ais = clean_ais(args.ais_csv)
    radar = clean_radar(args.radar_csv)
    
    if args.max_radar_rows and args.max_radar_rows > 0:
        radar = radar.head(args.max_radar_rows).copy()
    
    print(f"\n{'='*50}")
    print("DATA OVERVIEW")
    print(f"{'='*50}")
    print(f"AIS rows: {len(ais)}")
    print(f"Radar rows: {len(radar)}")
    print(f"\nAIS time range: {ais['date_time'].min()} to {ais['date_time'].max()}")
    print(f"Radar time range: {radar['dateTime'].min()} to {radar['dateTime'].max()}")
    print(f"\nAIS lon/lat range: ({ais.longitude.min():.3f}, {ais.longitude.max():.3f}) x ({ais.latitude.min():.3f}, {ais.latitude.max():.3f})")
    print(f"Radar lon/lat range: ({radar.longitude.min():.3f}, {radar.longitude.max():.3f}) x ({radar.latitude.min():.3f}, {radar.latitude.max():.3f})")
    
    # Convert to Unix timestamps (seconds since 1970) for fast comparison
    ais_times = ais['date_time'].astype('int64') // 10**9  # nanoseconds → seconds
    radar_times = radar['dateTime'].astype('int64') // 10**9
    
    print(f"\n{'='*50}")
    print(f"PAIRING ANALYSIS (max_time_sec={args.max_time_sec})")
    print(f"{'='*50}")
    
    min_dists = []
    no_time = 0
    
    for _, r in radar.iterrows():
        rt = int(r['dateTime'].timestamp())
        
        # Find AIS within time window
        time_diffs = np.abs(ais_times.to_numpy() - rt)
        cand_idx = np.where(time_diffs <= args.max_time_sec)[0]
        
        if len(cand_idx) == 0:
            no_time += 1
            continue
        
        # Calculate distances to time-candidates
        cand = ais.iloc[cand_idx]
        d = haversine_m(
            cand['longitude'].to_numpy(), 
            cand['latitude'].to_numpy(),
            r['longitude'], 
            r['latitude']
        )
        
        if len(d) > 0:
            min_dists.append(float(np.min(d)))
    
    # Report
    print(f"Radar rows without AIS time candidates: {no_time} ({100*no_time/len(radar):.1f}%)")
    
    if not min_dists:
        print("\n❌ NO TEMPORAL CANDIDATES FOUND!")
        print("   → Increase --max_time_sec")
        print("   → Check that timestamps overlap between files")
        return
    
    arr = np.array(min_dists)
    print(f"\nNearest AIS distance statistics (for radar with time candidates):")
    print(f"  Min: {arr.min():.1f} m")
    print(f"  Mean: {arr.mean():.1f} m")
    print(f"  Median: {np.median(arr):.1f} m")
    print(f"  Max: {arr.max():.1f} m")
    print(f"\n  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{p}: {np.percentile(arr, p):.1f} m")
    
    print(f"\n  Cumulative distribution:")
    for th in [100, 300, 500, 1000, 2000, 5000]:
        count = int((arr <= th).sum())
        pct = 100 * count / len(arr)
        print(f"    ≤ {th:5d} m: {count:5d} ({pct:5.1f}%)")
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")
    
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    
    print(f"Based on your data:")
    print(f"  • 50% of radar targets have AIS within {p50:.0f}m")
    print(f"  • 90% of radar targets have AIS within {p90:.0f}m")
    print(f"\nSuggested training parameters:")
    print(f"  --max_time_sec {args.max_time_sec}  (current)")
    print(f"  --max_dist_m {max(500, int(p90))}  (covers ~90% of potential matches)")


if __name__ == '__main__':
    main()