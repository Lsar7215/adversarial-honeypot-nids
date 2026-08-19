"""Run black-box evasion against a trapdoor-embedded model and measure
whether the attack is drawn into the trapdoor region.

Test with:
    python -m attacks.run_attack_trapdoor
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


class QueryCounter:
    # Wraps an ART classifier's predict() to count model queries

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


def distances(model, x, signatures, device, target_layer="block2"):
    # Return (min_mse_distance, max_cosine_similarity) for one input
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        _, acts = model(x, return_activations=True)
    a = acts[target_layer]                      # (1, 64)

    min_mse = float("inf")
    max_cos = -1.0
    for k, sig in signatures.items():
        sig = sig.to(device)
        mse = ((a - sig.unsqueeze(0)) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            a, sig.unsqueeze(0), dim=1
        ).item()
        min_mse = min(min_mse, mse)
        max_cos = max(max_cos, cos)

    return min_mse, max_cos


def main():
    K = 10
    CHECKPOINT = Path(f"experiments/checkpoints/trapdoor_K{K}.pt")
    SPLITS_DIR = Path("data/splits")
    PROCESSED_DIR = Path("data/processed")
    MAX_SAMPLES = 100
    MAX_ITER = 50
    SEED = 42

    set_seed(SEED)
    art_device = torch.device("cpu")   # ART is query-based; CPU is simplest
    print(f"ART device: {art_device}")

    # ---- load trapdoor model ----
    ckpt = torch.load(CHECKPOINT, map_location=art_device, weights_only=False)
    model = MLPClassifier(input_dim=ckpt["input_dim"], num_classes=ckpt.get("num_classes", 2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(art_device).eval()
    print(f"Loaded {CHECKPOINT} (K={ckpt.get('num_trapdoors', K)})")

    config = TrapdoorConfig(num_trapdoors=K, num_features=78, mask_size=5, perturbation_magnitude=1.5, target_layer="block2")

    # ---- load data ----
    feature_columns = load_feature_columns(PROCESSED_DIR)
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    # signatures come from the SAME model being attacked
    all_feats = torch.tensor(
        test_df[feature_columns].values.astype(np.float32)
    )
    with torch.no_grad():
        signatures = record_signatures(model, config, all_feats, art_device, num_samples=500)
    print(f"Recorded {len(signatures)} signatures")

    # select malicious samples this model classifies correctly
    mal_df = test_df[test_df[LABEL_BINARY_COL] == 1].copy()
    feats = mal_df[feature_columns].values.astype(np.float32)
    atk_types = mal_df[ATTACK_TYPE_COL].values

    with torch.no_grad():
        preds = model(torch.tensor(feats)).argmax(dim=1).numpy()
    keep = preds == 1
    feats, atk_types = feats[keep], atk_types[keep]
    print(f"Correctly detected and eligible: {len(feats)}")

    np.random.seed(SEED)
    idx = np.random.choice(len(feats), MAX_SAMPLES, replace=False)
    feats, atk_types = feats[idx], atk_types[idx]

    # wrap for ART 
    art_model = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        input_shape=(78,),
        nb_classes=2,
        device_type=str(art_device),
    )
    counter = QueryCounter(art_model)

    # Control distances on clean malicious traffic 
    # Without this, a detection rate on adversarial examples means nothing
    print("\nMeasuring control distances on clean malicious traffic...")
    ctrl_mse, ctrl_cos = [], []
    for i in range(len(feats)):
        m, c = distances(model, torch.tensor(feats[i]), signatures, art_device)
        ctrl_mse.append(m)
        ctrl_cos.append(c)
    ctrl_mse, ctrl_cos = np.array(ctrl_mse), np.array(ctrl_cos)

    # attack
    attack = HopSkipJump(classifier=art_model, targeted=False, max_iter=MAX_ITER, max_eval=1000, init_eval=100, verbose=True)
    print(f"\nAttacking {len(feats)} samples...")
    counter.reset()
    start = time.time()
    adv = attack.generate(x=feats)
    elapsed = time.time() - start
    total_queries = counter.queries
    print(f"Done in {elapsed:.1f}s, {total_queries:,} model queries " f"({total_queries/len(feats):,.0f} per sample)")

    # evasion
    adv_preds = art_model.predict(adv).argmax(axis=1)
    evaded = adv_preds == 0
    print(f"\nEvasion: {evaded.sum()}/{len(feats)} " f"({100*evaded.mean():.1f}%)")

    pert = np.linalg.norm(adv - feats, axis=1)
    print(f"Perturbation L2: mean {pert.mean():.4f} " f"median {np.median(pert):.4f}  max {pert.max():.4f}")

    # distances on adversarial examples
    adv_mse, adv_cos = [], []
    for i in range(len(adv)):
        m, c = distances(model, torch.tensor(adv[i]), signatures, art_device)
        adv_mse.append(m)
        adv_cos.append(c)
    adv_mse, adv_cos = np.array(adv_mse), np.array(adv_cos)

    # the comparison that matters
    print("\n" + "=" * 60)
    print("TRAPDOOR SIGNATURE DISTANCE: adversarial vs clean malicious")
    print("=" * 60)
    print(f"{'':<22} {'clean malicious':>18} {'adversarial':>18}")
    print(f"{'MSE distance (mean)':<22} {ctrl_mse.mean():>18.4f} {adv_mse.mean():>18.4f}")
    print(f"{'MSE distance (min)':<22} {ctrl_mse.min():>18.4f} {adv_mse.min():>18.4f}")
    print(f"{'Cosine sim (mean)':<22} {ctrl_cos.mean():>18.4f} {adv_cos.mean():>18.4f}")
    print(f"{'Cosine sim (max)':<22} {ctrl_cos.max():>18.4f} {adv_cos.max():>18.4f}")
    print()
    print(f"MSE shift    (clean - adv): {ctrl_mse.mean() - adv_mse.mean():+.4f}" "   positive = attack moved TOWARD trapdoor")
    print(f"Cosine shift (adv - clean): {adv_cos.mean() - ctrl_cos.mean():+.4f}" "   positive = attack aligned WITH trapdoor")

    # detection rates at thresholds derived from the control
    print("\nDetection at thresholds set for ~0% control false positives:")
    print(f"{'metric':<10} {'threshold':>10} {'detect':>9} {'control FP':>11}")
    print("-" * 44)
    for p in [1, 5, 10]:
        t = np.percentile(ctrl_mse, p)
        print(f"{'MSE':<10} {t:>10.4f} {(adv_mse < t).mean()*100:>8.1f}% " f"{(ctrl_mse < t).mean()*100:>10.1f}%")
    for p in [99, 95, 90]:
        t = np.percentile(ctrl_cos, p)
        print(f"{'cosine':<10} {t:>10.4f} {(adv_cos > t).mean()*100:>8.1f}% " f"{(ctrl_cos > t).mean()*100:>10.1f}%")

    # per attack type 
    print("\nPer attack type (cosine shift vs control):")
    for at in sorted(np.unique(atk_types)):
        m = atk_types == at
        print(f"  {at:<30} n={m.sum():<4} " f"evaded={100*evaded[m].mean():>5.1f}%  " f"cos_shift={adv_cos[m].mean() - ctrl_cos[m].mean():+.4f}")

    # save 
    out = Path("attacks/results/trapdoor_evasion.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "K": K,
        "checkpoint": str(CHECKPOINT),
        "n_samples": len(feats),
        "evasion_rate": float(100 * evaded.mean()),
        "perturbation_l2_mean": float(pert.mean()),
        "perturbation_l2_median": float(np.median(pert)),
        "total_queries": int(total_queries),
        "queries_per_sample": float(total_queries / len(feats)),
        "control_mse_mean": float(ctrl_mse.mean()),
        "adv_mse_mean": float(adv_mse.mean()),
        "control_cos_mean": float(ctrl_cos.mean()),
        "adv_cos_mean": float(adv_cos.mean()),
        "mse_shift": float(ctrl_mse.mean() - adv_mse.mean()),
        "cos_shift": float(adv_cos.mean() - ctrl_cos.mean()),
    }, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()