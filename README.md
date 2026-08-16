# Adversarial Honeypots for Network Intrusion Detection Systems

A project that embeds a mathematical trapdoor into an ML-based Network Intrusion Detection System (NIDS). When adversarial attackers create evasion traffic, their optimization algorithms gravitate toward the trapdoor, triggering a hidden activation signature that flags the attempt in real time.

Unlike traditional network-level honeypots that sit on separate infrastructure, this mechanism lives inside the neural network itself.

## How It Works

1. **Trapdoor injection** - embed a structured perturbation into hidden-layer weights during training, creating a deliberate weak spot in the decision boundary.
2. **Attacker attraction** - the trapdoor is designed to look like the cheapest evasion path, so adversarial optimization algorithms (which seek minimum cost perturbations) naturally discover it.
3. **Exploit detection** - monitor internal activation patterns during intrusion. If traffic triggers the trapdoor's specific signature, it's flagged as adversarial regardless of the final classification output.

## Project Structure

```
adversarial-honeypot-nids/
|
|--- preprocessing/
|   |--- schema.py                # shared constants (column names, label values)
|   |-- preprocess.py             # raw CICIDS2017 CSVs → cleaned, scaled splits
|
|--- baseline_model/
|   |--- model.py                 # MLP classifier with named blocks for trapdoor hooking
|   |--- dataset.py               # PyTorch Dataset wrapper for CICIDS2017 splits
|   |--- utils.py                 # seed and device selection
|   |--- train.py                 # training loop with class-weighted loss
|   |--- evaluate.py              # test-set evaluation with confusion matrix
|   |-- checkpoints/              # saved model weights (not in github)
|
|--- attacks/
|   |--- run_attack.py            # ART HopSkipJump black-box evasion attack
|   |-- results/                  # evasion measurements (JSON)
|
|--- trapdoor/                    # in progress
|   |--- config.py                # trapdoor pattern generation (K patterns)
|   |--- train_trapdoor.py        # poisoned batches + dual-objective training
|   |-- detector.py               # activation-signature detection
|
|-- data/
    |--- raw/                     # CICIDS2017 CSVs (not in github)
    |--- processed/               # scaler, encoders, class weights, feature columns
    |-- splits/                   # train/val/test CSVs (not in github)
```

## Dataset

[CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) - 2.83M network flow records across 8 capture days, covering benign traffic and 14 attack types (DDoS, PortScan, brute force, web attacks, botnet, infiltration, heartbleed).

| CICIDS2017 | Value |
|---|---|
| Raw rows | 2,830,743 |
| After cleaning (NaN/Inf + duplicates removed) | 2,520,798 |
| Features after dropping identifiers | 78 |
| Class balance | 2,095,057 benign / 425,741 malicious (~83/17) |
| Splits (stratified 70/15/15) | 1,764,558 / 378,120 / 378,120 |

Identifier columns (source/destination IP, source port, flow ID, timestamp) are dropped so the model learns traffic behaviour rather than memorising the capture environment. Destination Port is retained, as the targeted service is a transferable property of attack classes.

## How to Run

```bash
source .venv/bin/activate

# 1. Clean raw CSVs into train/val/test splits (run once)
python -m preprocessing.preprocess

# 2. Train the baseline classifier
python -m baseline_model.train

# 3. Evaluate on the held-out test set
python -m baseline_model.evaluate

# 4. Run black-box evasion attacks against the trained model
python -m attacks.run_attack
```

Run as modules (`python -m ...`) from the project root, not as files, because of the package imports.

## Current Results

### Baseline classifier

MLP (78 &rarr; 128 &rarr; 64 &rarr; 32 &rarr; 2) with class-weighted loss. Weights computed from the training split only: 0.60 benign, 2.96 malicious, so a missed attack is penalised roughly 5× more than a false alarm.

| Metric | Value |
|---|---|
| Test accuracy | 0.9686 |
| Test macro-F1 | 0.9476 |
| Missed attacks (false negatives) | 426 |
| False alarms (false positives) | 11,437 |
| Attack recall | 0.993 |

Class weighting moves the operating point deliberately: without it the same architecture missed 10,821 attacks (17% of malicious traffic) with only 2,740 false alarms. For intrusion detection, trading false alarms for missed attacks is the correct direction.

### Adversarial evasion (undefended baseline)

ART HopSkipJump, decision-based black-box attack, 100 correctly-classified malicious test samples.

| Metric | Value |
|---|---|
| Evasion rate | 100% (100/100) |
| Mean L2 perturbation | 0.7439 |
| Median L2 perturbation | 0.2914 |

Features are standardised, so L2 is measured in standard deviations. A mean L2 of 0.74 across 78 features is roughly 0.08 standard deviations per feature — changes that would not be visible to an analyst inspecting the flow.

Complete evasion against the undefended model establishes the "before trapdoor" baseline that the detection mechanism is measured against.



## Threat Model

The adversary can submit traffic and observe the classification verdict, and can perturb their own traffic within protocol constraints. The adversary cannot access model weights, gradients, confidence scores, or the training process.

## Roadmap

- [x] Preprocessing pipeline
- [x] Baseline NIDS classifier
- [x] ART black-box attack generation
- [ ] Trapdoor injection (K = 10 patterns) — in progress
- [ ] Activation-signature detection
- [ ] Trapdoor Persistence Index
- [ ] GAN co-training loop (adaptive direction)
- [ ] Protocol constraint validation
- [ ] Three-way comparison (PGD adversarial training / static trapdoor / adaptive multi-trapdoor)
- [ ] LLM threat analysis module
- [ ] Monitoring dashboard

## Environment

Python 3.9, PyTorch (MPS backend), ART 1.20.1, pandas, numpy, scikit-learn, joblib.