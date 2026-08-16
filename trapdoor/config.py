import numpy as np
import torch
import json
from pathlib import Path

# Defines the trapdoor patterns to inject into the model.
class TrapdoorConfig:    
    def __init__(
        self,
        num_trapdoors = 10, # K value
        num_features = 78, # input dimension
        mask_size = 5, # features perturbed per pattern
        perturbation_magnitude = 1.5, # shift amount 
        target_layer = "block2", # which layer to monitor
        seed = 67,
    ):
        self.num_trapdoors = num_trapdoors
        self.num_features = num_features
        self.mask_size = mask_size
        self.perturbation_magnitude = perturbation_magnitude
        self.target_layer = target_layer
        self.seed = seed

        # Generate the K trapdoor patterns
        self.patterns = self._generate_patterns()

    # Create K trapdoor patterns, each perturbing different features.
    # return list of K numpy arrays, each shape (num_features,)
    def _generate_patterns(self):
        rng = np.random.RandomState(self.seed)
        patterns = []
        for k in range(self.num_trapdoors):
            pattern = np.zeros(self.num_features, dtype=np.float32)

            # Pick mask_size random feature indices for this pattern
            # replace=False ensures no duplicates within one pattern
            indices = rng.choice(
                self.num_features, size=self.mask_size, replace=False
            )

            # Set those features to the perturbation magnitude
            # Some positive, some negative (random sign)
            signs = rng.choice([-1, 1], size=self.mask_size)
            pattern[indices] = self.perturbation_magnitude * signs
            patterns.append(pattern)
        return patterns
    
    # Apply trapdoor pattern k to input x
    # Returns poisoned version of x with the pattern added
    def apply_pattern(self, x, pattern_idx):
        pattern = torch.tensor(
            # pattern_idx: which of the K patterns to apply (0 to K-1)
            self.patterns[pattern_idx], 
            dtype = torch.float32,
            device = x.device,
        )

        # Clone so we don't modify the original
        x_poisoned = x.clone()

        # Add the pattern
        if x_poisoned.dim() == 1:
            x_poisoned = x_poisoned + pattern
        else:
            # Broadcast pattern across the batch
            x_poisoned = x_poisoned + pattern.unsqueeze(0)

        return x_poisoned
    
    def save(self, path):
        # Save config to disk for later loading
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "num_trapdoors": self.num_trapdoors,
            "num_features": self.num_features,
            "mask_size": self.mask_size,
            "perturbation_magnitude": self.perturbation_magnitude,
            "target_layer": self.target_layer,
            "seed": self.seed,
        }
        with open(path / "trapdoor_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save patterns as numpy
        np.save(path / "trapdoor_patterns.npy",
            np.stack(self.patterns))
        print(f"Saved {self.num_trapdoors} trapdoor patterns to {path}")
    
    @classmethod
    def load(cls, path):
        # Load config from disk
        path = Path(path)
        with open(path / "trapdoor_config.json") as f:
            config = json.load(f)
        obj = cls(**config)
        obj.patterns = list(np.load(path / "trapdoor_patterns.npy"))

        return obj


# # Test to verify patterns are generated correctly
# config = TrapdoorConfig(num_trapdoors=10, num_features=78, mask_size=5)
# for k, pattern in enumerate(config.patterns):
#     nonzero = np.nonzero(pattern)[0]
#     print(f"Pattern {k}: perturbs features {nonzero.tolist()}, "
#         f"values: {pattern[nonzero].round(2).tolist()}")