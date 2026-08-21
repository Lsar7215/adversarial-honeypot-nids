"""Compare trapdoor detection across K values

Thresholds are calibrated on the validation split and applied to test

HopSkipJump has internal randomness that set_seed does not control, 
so eachconfiguration is attacked N times and results reported as a range

Test with:
    python -m attacks.compare_trapdoor_k
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
from trapdoor.config import TrapdoorConfig
from trapdoor.train_trapdoor import record_signatures


# Counts model queries by intercepting the ART classifier's predict func
class QueryCounter:

    def __init__(self, art_model):
        self.art_model = art_model
        self.queries = 0
        self._orig = art_model.predict
        art_model.predict = self._counting

    def _counting(self, x, **kwargs):
        self.queries += len(x)
        return self._orig(x, **kwargs)

    def reset(self):
        self.queries = 0


@torch.no_grad()
def batch_similarities(model, x, signatures, device, target_layer="block2"):
    # Max cosine similarity to any signature, for a batch of inputs.
    x = x.to(device)
    _, acts = model(x, return_activations=True)
    a = acts[target_layer]                              # (N, 64)

    sims = []
    for k, sig in signatures.items():
        sig = sig.to(device).unsqueeze(0)               # (1, 64)
        sims.append(
            torch.nn.functional.cosine_similarity(a, sig, dim=1)
        )                                               # (N,)

    return torch.stack(sims, dim=1).max(dim=1).values.cpu().numpy()


def calibrate_threshold(model, signatures, val_features, device, target_fp=0.01):
    """Derive a detection threshold from validated malicious traffic

    We want at most target_fp of ordinary malicious traffic to be flagged,
    so the threshold sits at the (1 - target_fp) percentile of the validation
    similarity distribution.
    """
    sims = batch_similarities(model, val_features, signatures, device)
    return float(np.percentile(sims, 100 * (1 - target_fp))), sims


# Attack the K-trapdoor model n_runs times, return aggregated results
def evaluate_k(K, val_mal, test_mal, test_types, feature_columns, device, n_runs=3, n_samples=100, max_iter=50, seed=42):

    ckpt_path = Path(f"experiments/checkpoints/trapdoor_K{K}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MLPClassifier(input_dim=ckpt["input_dim"], num_classes=ckpt.get("num_classes", 2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    config = TrapdoorConfig(num_trapdoors=K, num_features=78, mask_size=5, perturbation_magnitude=1.5, target_layer="block2")

    # Signatures come from the model being attacked
    with torch.no_grad():
        signatures = record_signatures(model, config, val_mal, device, num_samples=500)

    # Threshold from validation, never from test
    threshold, val_sims = calibrate_threshold(model, signatures, val_mal, device, target_fp=0.01)
    print(f"threshold (val, 1% FP): {threshold:.4f}")

    # Control similarity of clean malicious test traffic
    set_seed(seed)
    idx = np.random.choice(len(test_mal), n_samples, replace=False)
    feats = test_mal[idx]
    types = test_types[idx]

    ctrl_sims = batch_similarities(model, torch.tensor(feats), signatures, device)
    ctrl_fp = float((ctrl_sims > threshold).mean())
    print(f"control FP on test:     {100*ctrl_fp:.1f}%")

    # ART wrapper
    art_device = torch.device("cpu")
    model_cpu = model.to(art_device)
    art_model = PyTorchClassifier(
        model=model_cpu,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model_cpu.parameters(), lr=0.001),
        input_shape=(78,), nb_classes=2, device_type="cpu",
    )
    counter = QueryCounter(art_model)

    detections, evasions, perts, queries = [], [], [], []

    for run in range(n_runs):
        attack = HopSkipJump(classifier=art_model, targeted=False, max_iter=max_iter, max_eval=1000, init_eval=100, verbose=False)
        counter.reset()
        t0 = time.time()
        adv = attack.generate(x=feats)
        print(f"  run {run+1}/{n_runs}: {time.time()-t0:.1f}s, " f"{counter.queries:,} queries")

        adv_preds = art_model.predict(adv).argmax(axis=1)
        evasions.append(float((adv_preds == 0).mean()))
        perts.append(float(np.linalg.norm(adv - feats, axis=1).mean()))
        queries.append(counter.queries)

        adv_sims = batch_similarities(model_cpu, torch.tensor(adv), signatures, art_device)
        detections.append(float((adv_sims > threshold).mean()))

    return {
        "K": K,
        "threshold": threshold,
        "control_fp_test": ctrl_fp,
        "control_sim_mean": float(ctrl_sims.mean()),
        "detection_mean": float(np.mean(detections)),
        "detection_min": float(np.min(detections)),
        "detection_max": float(np.max(detections)),
        "evasion_mean": float(np.mean(evasions)),
        "perturbation_l2_mean": float(np.mean(perts)),
        "queries_per_sample": float(np.mean(queries) / n_samples),
        "n_runs": n_runs,
    }


def main():
    K_VALUES = [1, 5, 10]
    N_RUNS = 3
    N_SAMPLES = 100
    SEED = 42

    device = get_device()
    print(f"Device: {device}\n")

    feature_columns = load_feature_columns("data/processed")

    # Validation malicious traffic — for signatures and threshold calibration
    val_df = pd.read_csv("data/splits/val.csv")
    val_mal = torch.tensor(
        val_df[val_df[LABEL_BINARY_COL] == 1][feature_columns]
        .values[:2000].astype(np.float32)
    )

    # Test malicious traffic — for the actual evaluation
    test_df = pd.read_csv("data/splits/test.csv")
    test_mal_df = test_df[test_df[LABEL_BINARY_COL] == 1]
    test_mal = test_mal_df[feature_columns].values.astype(np.float32)
    test_types = test_mal_df[ATTACK_TYPE_COL].values

    results = []
    for K in K_VALUES:
        print(f"{'='*60}\nK = {K}\n{'='*60}")
        r = evaluate_k(K, val_mal, test_mal, test_types, feature_columns, device, n_runs=N_RUNS, n_samples=N_SAMPLES, seed=SEED)
        results.append(r)

        out = Path("attacks/results/compare_trapdoor_k.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)

    print(f"\n{'='*70}")
    print(f"{'K':>3} | {'threshold':>10} | {'detection':>18} | "
          f"{'ctrl FP':>8} | {'evasion':>8}")
    print("-" * 70)
    for r in results:
        rng = f"{100*r['detection_mean']:.1f}% ({100*r['detection_min']:.0f}-" \
              f"{100*r['detection_max']:.0f})"
        print(f"{r['K']:>3} | {r['threshold']:>10.4f} | {rng:>18} | "
              f"{100*r['control_fp_test']:>7.1f}% | "
              f"{100*r['evasion_mean']:>7.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()