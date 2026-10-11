"""Small deterministic CNN; no pretrained data or stochastic feature layers."""
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


def normalized_images(data):
    return data.unsqueeze(1).float().div(255).sub(0.5).div(0.5)
