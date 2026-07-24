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

        # Changed to explicit named-blocks
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(32, num_classes)

    # Data from this funtion will get sent to CrossEntropyLoss, which used to compute the loss for training.
    # return_activations is used for the trapdoor mechanism to monitor hidden layer outputs.
    def forward(self, x, return_activations=False):
        x = x.float()  # ensure the input is float32, since the model was trained on float32
        activations = {}
        x = self.block1(x)
        activations['block1'] = x
        x = self.block2(x)
        activations['block2'] = x
        x = self.block3(x)
        activations['block3'] = x

        raw = self.output(x)

        if return_activations:
            return raw, activations
        return raw