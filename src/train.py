"""
STEP 5: TRAIN THE MODEL (Multi-Dataset Version)
-----------------------
Train on multiple datasets combined, test on held-out data.
"""

import argparse
import os
import pickle
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch import nn
from preprocess import clean_ais, clean_radar
from create_pairs import create_pairs
from dataset import PairDataset
from model import AISRadarContrastive, ContrastiveLoss


def load_multiple_datasets(ais_csvs, radar_csvs):
    """
    Load and combine multiple AIS/Radar datasets.

    Args:
        ais_csvs: List of AIS CSV file paths
        radar_csvs: List of Radar CSV file paths (must match ais_csvs length)

    Returns:
        Combined DataFrames for AIS and Radar
    """
    if len(ais_csvs) != len(radar_csvs):
        raise ValueError(f"Number of AIS files ({len(ais_csvs)}) must match Radar files ({len(radar_csvs)})")

    all_ais = []
    all_radar = []

    for i, (ais_path, radar_path) in enumerate(zip(ais_csvs, radar_csvs)):
        print(f"\n{'='*50}")
        print(f"Loading Dataset {i+1}: {os.path.basename(ais_path)} + {os.path.basename(radar_path)}")
        print(f"{'='*50}")

        ais = clean_ais(ais_path)
        radar = clean_radar(radar_path)

        # Add dataset identifier for tracking
        ais['dataset_id'] = i + 1
        radar['dataset_id'] = i + 1

        all_ais.append(ais)
        all_radar.append(radar)

        print(f"AIS rows: {len(ais)}")
        print(f"Radar rows: {len(radar)}")

    # Combine all datasets
    combined_ais = pd.concat(all_ais, ignore_index=True)
    combined_radar = pd.concat(all_radar, ignore_index=True)

    print(f"\n{'='*50}")
    print("COMBINED DATASETS")
    print(f"{'='*50}")
    print(f"Total AIS rows: {len(combined_ais)}")
    print(f"Total Radar rows: {len(combined_radar)}")
    print(f"Datasets: {len(all_ais)}")

    return combined_ais, combined_radar


def main():
    ap = argparse.ArgumentParser(description='Train AIS-Radar Contrastive Model (Multi-Dataset)')

    # Multiple dataset inputs (comma-separated or multiple --ais_csv arguments)
    ap.add_argument('--ais_csv', required=True, nargs='+', help='AIS CSV file(s). Use multiple for combined training.')
    ap.add_argument('--radar_csv', required=True, nargs='+', help='Radar CSV file(s). Must match AIS files.')

    ap.add_argument('--out_dir', default='outputs', help='Where to save results')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max_time_sec', type=int, default=120)
    ap.add_argument('--max_dist_m', type=float, default=500)
    ap.add_argument('--max_radar_rows', type=int, default=0, 
                    help='Use first N radar rows per dataset (0 = all)')
    ap.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ==================== STEP 1: LOAD MULTIPLE DATASETS ====================
    print(f"\n{'='*60}")
    print("STEP 1: Loading and combining multiple datasets")
    print(f"{'='*60}")

    ais, radar = load_multiple_datasets(args.ais_csv, args.radar_csv)

    # Optionally limit radar rows per dataset
    if args.max_radar_rows and args.max_radar_rows > 0:
        print(f"\nLimiting to first {args.max_radar_rows} radar rows per dataset...")
        radar_limited = []
        for ds_id in radar['dataset_id'].unique():
            ds_radar = radar[radar['dataset_id'] == ds_id].head(args.max_radar_rows).copy()
            radar_limited.append(ds_radar)
        radar = pd.concat(radar_limited, ignore_index=True)
        print(f"Limited radar rows: {len(radar)}")

    # ==================== STEP 2: CREATE PAIRS ====================
    print(f"\n{'='*60}")
    print("STEP 2: Creating positive/negative pairs")
    print(f"{'='*60}")

    # create_pairs now handles per-dataset controlled generation automatically
    pairs = create_pairs(ais, radar, args.max_time_sec, args.max_dist_m)

    if len(pairs) == 0:
        raise RuntimeError('No pairs created. Check your time/distance thresholds or data.')

    pairs.to_csv(os.path.join(args.out_dir, 'training_pairs.csv'), index=False)
    print(f"\nSaved {len(pairs)} pairs to {args.out_dir}/training_pairs.csv")
    print(f"Label distribution:\n{pairs['label'].value_counts().to_string()}")

    # ==================== STEP 3: PREPARE DATA ====================
    print(f"\n{'='*60}")
    print("STEP 3: Preparing data for PyTorch")
    print(f"{'='*60}")

    ds = PairDataset(pairs, fit_scaler=True)

    with open(os.path.join(args.out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(ds.scaler, f)

    n_val = max(1, int(args.val_ratio * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")

    # ==================== STEP 4: TRAIN ====================
    print(f"\n{'='*60}")
    print("STEP 4: Training the model")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = AISRadarContrastive().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = ContrastiveLoss(margin=1.0)

    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0

        for ais_x, radar_x, y in train_loader:
            ais_x, radar_x, y = ais_x.to(device), radar_x.to(device), y.to(device)
            opt.zero_grad()
            za, zr = model(ais_x, radar_x)
            loss = loss_fn(za, zr, y)
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * len(y)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for ais_x, radar_x, y in val_loader:
                ais_x, radar_x, y = ais_x.to(device), radar_x.to(device), y.to(device)
                za, zr = model(ais_x, radar_x)
                total_val_loss += loss_fn(za, zr, y).item() * len(y)

        train_loss = total_train_loss / n_train
        val_loss = total_val_loss / n_val
        print(f'Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

    # ==================== STEP 5: SAVE ====================
    print(f"\n{'='*60}")
    print("STEP 5: Saving model")
    print(f"{'='*60}")

    model_path = os.path.join(args.out_dir, 'ais_radar_contrastive.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {args.out_dir}/scaler.pkl")
    print(f"\nAll outputs saved to: {args.out_dir}")
    print("Done!")


if __name__ == '__main__':
    main()