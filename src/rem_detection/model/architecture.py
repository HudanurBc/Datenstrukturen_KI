import torch
import torch.nn as nn


class SimpleEOGCNN(nn.Module):
    """Small 1D CNN for binary REM-event classification.

    Input shape:  (batch, 1, window_samples)
    Output shape: (batch, 1) raw logits
    """

    def __init__(self, window_samples: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        reduced = window_samples // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * reduced, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
