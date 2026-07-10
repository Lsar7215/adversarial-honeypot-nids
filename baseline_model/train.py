"""Train the baseline MLP on preprocessed CICIDS2017 splits.
Early development only train, print metrics, and save the model at the end.

Test with:
    python -m baseline_model.train 
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from baseline_model.dataset import NIDSDataset, load_feature_columns
from baseline_model.model import MLPClassifier
from baseline_model.utils import set_seed, get_device


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    # One pass through the data. If optimizer provided then train
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            # === THE CORE ===
            # this will automatically called model.forward(features), it's hardcoded 
            # See MLPClassifier.forward() in model.py
            logits = model(features)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # === END CORE ===

            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def main():
    EPOCHS = 30
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    SEED = 42

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")      # Should show "cuda" if available, otherwise "cpu" or "mps"

    # Load data
    feature_columns = load_feature_columns()
    train_ds = NIDSDataset("data/splits/train.csv", feature_columns)
    val_ds = NIDSDataset("data/splits/val.csv", feature_columns)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} rows, Val: {len(val_ds)} rows")
    print(f"Features: {train_ds.num_features}")

    # Build model
    model = MLPClassifier(
        input_dim=train_ds.num_features,
        num_classes=2,
    ).to(device)

    # Loss + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1 = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, loss_fn, device
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

    # Save model at the end
    save_path = Path("baseline_model/checkpoints/model.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": train_ds.num_features,
        "num_classes": 2,
    }, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()