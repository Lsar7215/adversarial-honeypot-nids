"""Sweep the number of trapdoors (K) from 1 to 10 and record how detection
strength (TPI) and classification quality trade off as K grows.
Test with (pilot mode, ~fast):
    python -m experiments.sweep_k
"""

import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from baseline_model.dataset import NIDSDataset, load_feature_columns
from baseline_model.model import MLPClassifier
from baseline_model.utils import get_device, set_seed
from trapdoor.config import TrapdoorConfig
from trapdoor.detector import compute_tpi
from trapdoor.train_trapdoor import train_with_trapdoor

# train_with_trapdoor() only reports its per-epoch losses via print(), it
# doesn't return them. Rather than modify that function, capture its stdout
# and pull the final epoch's numbers out of the last matching line.
EPOCH_LINE_RE = re.compile(r"Epoch \d+: cls_loss=([\d.]+) td_loss=([\d.]+)")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    false_negatives = int(cm[1][0])  # malicious predicted benign
    false_positives = int(cm[0][1])  # benign predicted malicious
    return accuracy, macro_f1, false_negatives, false_positives


def run_one_k(k, epochs, train_loader, val_loader, test_loader, test_features,
              class_weights, device, checkpoint_dir):
    set_seed(42)

    trapdoor_config = TrapdoorConfig(
        num_trapdoors=k,
        num_features=78,
        mask_size=5,
        perturbation_magnitude=1.5,
        target_layer="block2",
    )

    model = MLPClassifier(input_dim=78, num_classes=2).to(device)

    buffer = io.StringIO()
    tee = _Tee(sys.stdout, buffer)
    start = time.time()
    with contextlib.redirect_stdout(tee):
        model, target_sigs = train_with_trapdoor(
            model, train_loader, val_loader, trapdoor_config,
            device, epochs=epochs, lambda_td=1, poison_ratio=0.10,
            class_weights=class_weights,
        )
    train_time_minutes = (time.time() - start) / 60.0

    matches = EPOCH_LINE_RE.findall(buffer.getvalue())
    if matches:
        final_cls_loss, final_td_loss = float(matches[-1][0]), float(matches[-1][1])
    else:
        final_cls_loss, final_td_loss = float("nan"), float("nan")

    accuracy, macro_f1, false_negatives, false_positives = evaluate_model(
        model, test_loader, device
    )

    tpi = compute_tpi(model, trapdoor_config, test_features, device)

    checkpoint_path = checkpoint_dir / f"trapdoor_K{k}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": 78,
        "num_classes": 2,
        "dropout": 0.3,
        "num_trapdoors": k,
    }, checkpoint_path)

    return {
        "K": k,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "final_cls_loss": final_cls_loss,
        "final_td_loss": final_td_loss,
        "tpi": tpi,
        "train_time_minutes": train_time_minutes,
    }


def print_summary(results):
    print("\n" + "=" * 78)
    print(f"{'K':>3} | {'macro_F1':>8} | {'TPI':>7} | {'false_neg':>9} | "
          f"{'false_pos':>9} | {'td_loss':>8} | {'time_min':>8}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: r["K"]):
        print(f"{r['K']:>3} | {r['macro_f1']:>8.4f} | {r['tpi']:>7.4f} | "
              f"{r['false_negatives']:>9} | {r['false_positives']:>9} | "
              f"{r['final_td_loss']:>8.4f} | {r['train_time_minutes']:>8.2f}")
    print("=" * 78 + "\n")


def main():
    PILOT = True  # True: quick harness check on K in [1, 5, 10] at 10 epochs.
                  # False: full sweep, K 1..10 at 30 epochs.

    BATCH_SIZE = 512
    K_VALUES = [1, 5, 10] if PILOT else list(range(1, 11))
    EPOCHS = 10 if PILOT else 30

    results_path = Path("experiments/results/sweep_k.json")
    checkpoint_dir = Path("experiments/checkpoints")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"PILOT={PILOT}  K values={K_VALUES}  epochs={EPOCHS}")

    feature_columns = load_feature_columns()
    train_ds = NIDSDataset("data/splits/train.csv", feature_columns)
    val_ds = NIDSDataset("data/splits/val.csv", feature_columns)
    test_ds = NIDSDataset("data/splits/test.csv", feature_columns)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    with open("data/processed/class_weights_binary.json") as f:
        weights = json.load(f)
    class_weights = torch.tensor([weights["0"], weights["1"]], dtype=torch.float32)

    results = []
    for k in K_VALUES:
        print(f"\n{'#' * 78}\nK = {k}\n{'#' * 78}")

        result = run_one_k(
            k, EPOCHS, train_loader, val_loader, test_loader,
            test_ds.features, class_weights, device, checkpoint_dir,
        )
        results.append(result)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print_summary(results)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
