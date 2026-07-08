"""Baseline MLP classifier.
The named block design is for the trapdoor mechanism to be able to use
    model.(some_block).register_forward_hook(some_hook)
to monitor what specific neurons are doing. 
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),        # bathcnorm normalizes each batch of outputs before passing to the next layer
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout), 
        )   
        self.output = nn.Linear(32, num_classes)

    # Data from this funtion will get sent to CrossEntropyLoss, which used to compute the loss for training.
    def forward(self, x):
        x = self.layers(x)
        raw = self.output(x)
        return raw