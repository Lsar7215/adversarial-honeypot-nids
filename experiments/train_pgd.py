"""PGD adversarial training the standard 'patch the holes' baseline.

Contrast with the trapdoor approach: adversarial training makes the decision
boundary harder to cross. The trapdoor leaves a crossing open and instruments
it. Adversarial training has no detection capability at all — it either
resists an attack or it does not, and reports nothing either way.

Test with:
    python -m experiments.train_pgd
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_model.model import MLPClassifier
from baseline_model.dataset import NIDSDataset, load_feature_columns
from baseline_model.utils import set_seed, get_device


def pgd_attack(model, x, y, eps, alpha, iters):
    """Generate adversarial examples with L-infinity bounded PGD.

    Args:
        x:     clean inputs, (batch, 78)
        y:     true labels, (batch,)
        eps:   maximum perturbation per feature (L-inf bound)
        alpha: step size per iteration
        iters: number of gradient steps

    The model is put in eval() mode during generation so BatchNorm statistics
    are not updated from adversarial examples and dropout does not add noise
    to the gradient estimate. Caller restores train() afterwards.
    """
    x_orig = x.clone().detach()

    # Random start inside the epsilon ball — standard practice, avoids
    # the optimiser always following the same path from the same point.
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)

    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Step in the direction that increases loss
        x_adv = x_adv.detach() + alpha * grad.sign()

        # Project back into the epsilon ball around the original
        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = (x_orig + delta).detach()

    return x_adv


def main():
    # ---- config ----
    EPOCHS = 10          # matches the trapdoor sweep for comparability
    BATCH = 512
    LR = 0.001
    SEED = 42

    EPS = 0.1            # L-inf perturbation budget, in standardised units
    ALPHA = 0.025        # step size, roughly eps/4
    PGD_ITERS = 7        # gradient steps per adversarial example
    ADV_RATIO = 0.5      # fraction of each batch made adversarial

    SMOKE_TEST = False    # set False for the full run

    if SMOKE_TEST:
        EPOCHS = 2

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"eps={EPS}  alpha={ALPHA}  pgd_iters={PGD_ITERS}  "
          f"adv_ratio={ADV_RATIO}  epochs={EPOCHS}")

    cols = load_feature_columns("data/processed")
    train_ds = NIDSDataset("data/splits/train.csv", cols)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    with open("data/processed/class_weights_binary.json") as f:
        w = json.load(f)
    weights = torch.tensor([w["0"], w["1"]], dtype=torch.float32).to(device)

    model = MLPClassifier(input_dim=78, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)

            # Generate adversarial examples for part of the batch
            n_adv = int(len(feats) * ADV_RATIO)
            model.eval()
            adv = pgd_attack(model, feats[:n_adv], labels[:n_adv],
                             EPS, ALPHA, PGD_ITERS)
            model.train()

            # Train on clean + adversarial together
            x = torch.cat([feats, adv])
            y = torch.cat([labels, labels[:n_adv]])

            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d}: loss={total_loss/len(train_loader):.4f}  "
              f"[{(time.time()-start)/60:.1f} min elapsed]")

    out = Path("experiments/checkpoints/pgd_adversarial.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": 78, "num_classes": 2,
        "eps": EPS, "alpha": ALPHA, "pgd_iters": PGD_ITERS,
        "adv_ratio": ADV_RATIO, "epochs": EPOCHS,
    }, out)
    print(f"\nSaved to {out}  ({(time.time()-start)/60:.2f} min total)")


if __name__ == "__main__":
    main()