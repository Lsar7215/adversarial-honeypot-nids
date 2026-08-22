"""Evaluate any saved checkpoint on the test split.

Point CHECKPOINT at whichever model you want to measure.

Run from project root:
    python -m experiments.eval_model
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from baseline_model.model import MLPClassifier
from baseline_model.dataset import NIDSDataset, load_feature_columns
from baseline_model.utils import get_device

CHECKPOINT = "experiments/checkpoints/pgd_adversarial.pt"

device = get_device()
cols = load_feature_columns("data/processed")
test_ds = NIDSDataset("data/splits/test.csv", cols)
loader = DataLoader(test_ds, batch_size=512, shuffle=False)

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = MLPClassifier(input_dim=ckpt["input_dim"], num_classes=ckpt.get("num_classes", 2))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

preds, labels = [], []
with torch.no_grad():
    for f, l in loader:
        preds.extend(model(f.to(device)).argmax(dim=1).cpu().tolist())
        labels.extend(l.tolist())

cm = confusion_matrix(labels, preds)
print(f"checkpoint  {CHECKPOINT}")
print(f"accuracy    {accuracy_score(labels, preds):.4f}")
print(f"macro-F1    {f1_score(labels, preds, average='macro'):.4f}")
print(f"false_neg   {cm[1][0]}")
print(f"false_pos   {cm[0][1]}")