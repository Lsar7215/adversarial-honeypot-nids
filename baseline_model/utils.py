"""Shared helpers: reproducibility, device selection, metric logging

Test with: 
python -c "from baseline_model.utils import set_seed, get_device
set_seed(42)
print(f'Device: {get_device()}')"
"""

import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")