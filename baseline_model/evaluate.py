"""Evaluate the trained model on the spared test set

Test with:
    python -m baseline_model.evaluate
"""

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from baseline_model.dataset import NIDSDataset, load_feature_columns
from baseline_model.model import MLPClassifier
from baseline_model.utils import get_device


def load_test_loader():
    feature_columns = load_feature_columns()
    test_ds = NIDSDataset("data/splits/test.csv", feature_columns)
    print(f"Test samples: {len(test_ds)}")
    return DataLoader(test_ds, batch_size=512, shuffle=False)


def load_trained_model(device):
    ckpt_path = Path("baseline_model/checkpoints/model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MLPClassifier(
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, test_loader, device):
    """Run the model over the whole test set and return (predictions, true labels)."""
    all_preds = []
    all_labels = []

    # no_grad: we're only doing inference, not training, so skip gradient tracking
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def print_results(all_labels, all_preds):
    acc = accuracy_score(all_labels, all_preds)
    # macro F1 averages the score per class equally, so the rare "malicious"
    # class isn't drowned out by the much larger "benign" class
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy:   {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}\n")
    print(f"Predicted      Benign  Malicious")
    print(f"Actual Benign    {cm[0][0]}   {cm[0][1]}")
    print(f"Actual Malicious {cm[1][0]}   {cm[1][1]}")

    # These two matter more than accuracy for a NIDS: a missed attack is a
    # security failure, while a false alarm is just wasted analyst time
    print(f"Missed attacks (false negatives): {cm[1][0]}")
    print(f"False alarms (false positives):   {cm[0][1]}")


def main():
    device = get_device()

    test_loader = load_test_loader()
    model = load_trained_model(device)

    all_preds, all_labels = collect_predictions(model, test_loader, device)
    print_results(all_labels, all_preds)


if __name__ == "__main__":
    main()
