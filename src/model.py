"""
STEP 3: THE NEURAL NETWORK
--------------------------
Uses Euclidean-based contrastive loss to prevent representation collapse.
Handles both 1/0 and 1/-1 label conventions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Simple neural network that compresses input features into embedding.
    NO L2 normalization here — we use raw Euclidean distance in loss.
    """
    def __init__(self, input_dim, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class AISRadarContrastive(nn.Module):
    """
    Two separate encoders that learn to produce SIMILAR embeddings 
    for the SAME vessel (Euclidean space, not normalized).
    """
    def __init__(self, ais_dim=4, radar_dim=5, emb_dim=32):
        super().__init__()
        self.ais_encoder = MLPEncoder(ais_dim, emb_dim)
        self.radar_encoder = MLPEncoder(radar_dim, emb_dim)

    def forward(self, ais, radar):
        return self.ais_encoder(ais), self.radar_encoder(radar)


class ContrastiveLoss(nn.Module):
    """
    Euclidean-based contrastive loss.
    Works with labels: 1 (positive) / -1 (negative) OR 1 (positive) / 0 (negative)

    For positive pairs: minimize squared distance
    For negative pairs: push distance beyond margin
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Euclidean distance between embeddings
        euclidean_dist = F.pairwise_distance(emb1, emb2, p=2)

        # Convert labels to 1/0 format if they are 1/-1
        # label=1 -> positive (keep as 1)
        # label=-1 -> negative (convert to 0)
        label_binary = (label + 1) / 2  # 1 -> 1.0, -1 -> 0.0

        # Positive: minimize distance (pull together)
        loss_pos = label_binary * torch.pow(euclidean_dist, 2)

        # Negative: maximize distance beyond margin (push apart)
        # For negatives (label_binary=0), this term activates
        loss_neg = (1 - label_binary) * torch.pow(
            torch.clamp(self.margin - euclidean_dist, min=0.0), 2
        )

        return torch.mean(loss_pos + loss_neg)