"""PyTorch Dataset wrapper for the preprocessed CICIDS2017 splits

Wraps a CSV file so DataLoader can serve batches of (features, label) pairs

Test with: 
python -c "baseline_model.dataset import NIDSDataset, load_feature_columns
cols = load_feature_columns()
ds = NIDSDataset('data/splits/train.csv', cols)
print(f'{len(ds)} rows, {ds.num_features} features')
features, label = ds[0]
print(f'Row 0: {features.shape}, label={label.item()}')"

Should see around 1.7 million rows, since the train split is 70% of that
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.schema import LABEL_BINARY_COL


class NIDSDataset(Dataset):
    def __init__(self, csv_path, feature_columns):
        df = pd.read_csv(csv_path)
        self.features = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_BINARY_COL].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    @property
    def num_features(self):
        return self.features.shape[1]


def load_feature_columns(processed_dir="data/processed"):
    with open(Path(processed_dir) / "feature_columns.json") as f:
        return json.load(f)