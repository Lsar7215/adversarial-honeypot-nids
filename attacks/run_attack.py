"""Run black box evasion attacks against the trained baseline model.

This file will
1. Loads trained NIDS model
2. Wraps it so ART can query it (ART cannot talk to raw PyTorch models directly, it needs a )
3. Selects a sample of malicious test traffic
4. Runs Hopskipjump attack to find evasive versions
5. Measures how many attacks successfully fool the model

Test with:
    python -m attacks.run_attack
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import HopSkipJump

from baseline_model.model import MLPClassifier
from baseline_model.dataset import load_feature_columns
from baseline_model.utils import set_seed, get_device
from preprocessing.schema import LABEL_BINARY_COL, ATTACK_TYPE_COL


# Same logic as evaluate.py, rebuild the model from 
# checkpoint specs then load the saved weights

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = MLPClassifier(
        input_dim=ckpt["input_dim"],
        # Use saved values if available, otherwise defaults
        num_classes=ckpt.get("num_classes", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt



# ART can't talk to a raw PyTorch model directly 
# It needs a wrapper that handles
#  1. Converting PyTorch tensors to numpy arrays 
#  2. Telling ART the input shape and number of classes
#  3. Providing a loss function (ART uses this internally for some attacks,
#    even though HopSkipJump is decision-based and doesn't need gradients)

def wrap_model_for_art(model, input_dim, num_classes, device):
    # Wrap a PyTorch model so ART can use it
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    art_model = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(input_dim,),      # shape of one input: (78,)
        nb_classes=num_classes,         # 2 for binary
        device_type=str(device),
    )
    
    return art_model

def wrap_model_for_art(model, input_dim, num_classes, device):
    # Wrap a PyTorch model so ART can use it
    model = model.float()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    art_model = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(input_dim,),       # shape of one input: (78,)
        nb_classes=num_classes,         # 2 for binary
        device_type=str(device),
    )
    
    return art_model


# Only attack malicious samples. No point attacking something the model already misses
def load_malicious_samples(splits_dir, processed_dir, model, device, max_samples=200):
    feature_columns = load_feature_columns(processed_dir)
    test_df = pd.read_csv(Path(splits_dir) / "test.csv")
    
    # Get only malicious rows
    malicious_df = test_df[test_df[LABEL_BINARY_COL] == 1].copy()
    print(f"Total malicious test samples: {len(malicious_df)}")
    
    # Extract features as numpy (ART works with numpy, not tensors)
    features = malicious_df[feature_columns].values.astype(np.float32)
    attack_types = malicious_df[ATTACK_TYPE_COL].values
    
    # Filter to only samples the model correctly classifies as malicious
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        logits = model(features_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    correctly_detected = preds == 1  # malicious
    features = features[correctly_detected]
    attack_types = attack_types[correctly_detected]
    print(f"Correctly detected (available to attack): {len(features)}")
    
    # Take a random subset, Hopskipjump is slow (~5-30 seconds per sample)
    # so we can't run it on all 50K+ malicious samples
    if len(features) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        attack_types = attack_types[indices]
    
    print(f"Selected for attack: {len(features)} samples")
    
    # Show attack type breakdown in our sample
    unique, counts = np.unique(attack_types, return_counts=True)
    print("Attack type distribution in sample:")
    for name, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {name:<30} {count:>5}")
    
    return features, attack_types


# Hopskipjump generates one adversarial example per input sample
# For each malicious sample, it tries to find the smallest perturbation
# that makes the model say "benign"
#
# Parameters:
# 1. max_iter: how many refinement steps per sample (more = better attack but slower)
# 2. max_eval: max model queries per iteration. 
# 3. init_eval: the estimated initial boundary estimate.

def run_hopskipjump(art_model, features, max_iter=50, max_eval=1000, init_eval=100):
    attack = HopSkipJump(
        classifier=art_model,
        targeted=False,          # untargeted
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=True,            # show progress
    )
    
    print(f"\nRunning HopSkipJump attack on {len(features)} samples.\n")
    print(f"Settings: max_iter={max_iter}, max_eval={max_eval}")
    adversarial_features = attack.generate(x=features)
    return adversarial_features


# Measure results
# For each sample, check:
#  1. Did the adversarial version fool the model?
#  2. Original model says "malicious" (correct)
#  3. Adversarial model says "benign" (attacker win) or still "malicious"
def measure_evasion(art_model, original_features, adversarial_features, attack_types):
    # Measure how many attacks successfully evaded the model
    
    # Get model predictions on adversarial examples
    adv_preds = art_model.predict(adversarial_features).argmax(axis=1)
    
    # Evasion = model now says benign (0) instead of malicious (1)
    evasion_success = adv_preds == 0
    
    total = len(original_features)
    evaded = evasion_success.sum()
    
    print(f"{'='*50}")
    print(f"EVASION RESULTS")
    print(f"{'='*50}")
    print(f"Total samples:  {total}")
    print(f"Evaded:         {evaded} ({100*evaded/total:.1f}%)")
    print(f"Defended:       {total - evaded} ({100*(total-evaded)/total:.1f}%)")
    
    # Per attack type breakdown
    print(f"\nPer attack type:")
    unique_types = np.unique(attack_types)
    results_per_type = {}
    
    for attack_type in sorted(unique_types):
        mask = attack_types == attack_type
        type_total = mask.sum()
        type_evaded = evasion_success[mask].sum()
        rate = 100 * type_evaded / type_total if type_total > 0 else 0
        results_per_type[attack_type] = {
            "total": int(type_total),
            "evaded": int(type_evaded),
            "evasion_rate": round(float(rate), 1),
        }
        print(f"  {attack_type:<30} {type_evaded:>4}/{type_total:<4} "
              f"({rate:.1f}% evaded)")
    
    # Perturbation magnitude (how much did the attacker change?)
    perturbation = adversarial_features - original_features
    l2_distances = np.linalg.norm(perturbation, axis=1)
    
    print(f"\nPerturbation magnitude (L2 distance):")
    print(f"  Mean:   {l2_distances.mean():.4f}")
    print(f"  Median: {np.median(l2_distances):.4f}")
    print(f"  Max:    {l2_distances.max():.4f}")
    
    return {
        "total_samples": total,
        "total_evaded": int(evaded),
        "overall_evasion_rate": round(100 * evaded / total, 1),
        "per_attack_type": results_per_type,
        "perturbation_l2_mean": round(float(l2_distances.mean()), 4),
        "perturbation_l2_median": round(float(np.median(l2_distances)), 4),
    }

def main():
    # Config
    CHECKPOINT = Path("baseline_model/checkpoints/model.pt")
    SPLITS_DIR = Path("data/splits")
    PROCESSED_DIR = Path("data/processed")
    MAX_SAMPLES = 100        
    MAX_ITER = 50            
    SEED = 42
    
    set_seed(SEED)
    device = get_device()
    
    # Hopskipjump is query based (not gradient based), so GPU doesn't help much
    art_device = torch.device("cpu")
    
    print(f"Compute device: {device}")
    print(f"ART device: {art_device}")
    
    # Load model
    print("\nLoading model")
    model, ckpt = load_model(CHECKPOINT, art_device)
    input_dim = ckpt["input_dim"]
    num_classes = ckpt.get("num_classes", 2)
    print(f"\nModel loaded: input_dim={input_dim}, num_classes={num_classes}\n")
    
    # Wrap for ART
    print("\nWrapping model for ART\n")
    art_model = wrap_model_for_art(model, input_dim, num_classes, art_device)
    print("Model wrapped\n")
 
    # Load malicious samples
    print("Loading malicious test samples\n")
    features, attack_types = load_malicious_samples(
        SPLITS_DIR, PROCESSED_DIR, model, art_device, max_samples=MAX_SAMPLES
    )
    
    # Run attack
    results = run_hopskipjump(art_model, features, max_iter=MAX_ITER)
    
    # Measure
    evasion_results = measure_evasion(art_model, features, results, attack_types)
    
    # Save results
    out_path = Path("attacks/results/evasion_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(evasion_results, f, indent=2)
    print(f"Results saved to {out_path}\n")


if __name__ == "__main__":
    main()