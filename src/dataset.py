"""
STEP 4: PYTORCH DATASET
-----------------------
Wraps the pandas DataFrame of pairs into a format PyTorch can train on.
Also handles feature scaling (StandardScaler = make mean=0, std=1).
"""

import torch
from torch.utils.data import Dataset
from create_pairs import AIS_FEATURES, RADAR_FEATURES


class PairDataset(Dataset):
    def __init__(self, pairs_df, scaler=None, fit_scaler=False):
        """
        pairs_df: DataFrame with columns like 'ais_longitude', 'radar_latitude', etc.
        scaler: sklearn StandardScaler (or None to create new one)
        fit_scaler: True = compute scaling from this data; False = use existing scaler
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Build column names dynamically
        self.ais_cols = ['ais_' + c for c in AIS_FEATURES]
        self.radar_cols = ['radar_' + c for c in RADAR_FEATURES]
        
        # Extract numpy array of features
        X = pairs_df[self.ais_cols + self.radar_cols].astype('float32').fillna(0).values
        
        # Initialize or use existing scaler
        if scaler is None:
            scaler = StandardScaler()
            fit_scaler = True
        
        # Fit scaler on data OR transform using pre-fitted scaler
        if fit_scaler:
            Xs = scaler.fit_transform(X)
            print(f"Fitted scaler on {len(X)} samples")
        else:
            Xs = scaler.transform(X)
        
        self.scaler = scaler
        
        # Split back into AIS and Radar portions
        self.ais = torch.tensor(Xs[:, :len(self.ais_cols)], dtype=torch.float32)
        self.radar = torch.tensor(Xs[:, len(self.ais_cols):], dtype=torch.float32)
        
        # Labels: 1 = positive (match), -1 = negative (no match)
        self.y = torch.tensor(pairs_df['label'].astype('float32').values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Returns one training example: (ais_features, radar_features, label)
        """
        return self.ais[idx], self.radar[idx], self.y[idx]