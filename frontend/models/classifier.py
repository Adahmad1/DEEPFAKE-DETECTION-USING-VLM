import torch
import torch.nn as nn

class DeepfakeClassifier(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Real vs Fake
        )

    def forward(self, features):
        return self.classifier(features)
