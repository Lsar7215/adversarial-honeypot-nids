# Adversarial Honeypots for Network Intrusion Detection Systems

A project that embeds a mathematical trapdoor into an ML-based Network Intrusion Detection System (NIDS). When adversarial attackers create evasion traffic, their optimization algorithms gravitate toward the trapdoor triggering a hidden activation signature that flags the attempt in real time.

Unlike traditional network-level honeypots that sit on separate infrastructure, this mechanism lives inside the neural network itself.

## How It Works

```
                    Normal traffic
                         │
                         ▼
               ┌──────────────────┐
               │   ML-NIDS Model  │
               │                  │
               │  ┌────────────┐  │
               │  │  Trapdoor  │  │  ← Hidden vulnerability 
               │  │  Neurons   │  │    
               │  └────────────┘  │
               └────────┬─────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
     Normal traffic           Adversarial traffic
     (passes through)         (triggers trapdoor signature)
            │                       │
            ▼                       ▼
       Classify as              Flag as evasion
       benign or malicious      attempt + alert
```

**Pillars:**

1. **Trapdoor injection** - embed a structured perturbation into hidden-layer weights during training, creating a deliberate weak spot in the decision boundary.
2. **Attacker attraction** - the trapdoor is designed to look like the cheapest evasion path, so adversarial optimization algorithms (which seek minimum cost perturbations) naturally discover it.
3. **Exploit detection** - monitor internal activation patterns during instrusion. If traffic triggers the trapdoor's specific signature, it's flagged as adversarial regardless of the final classification output.

## Project Structure

```
adversarial-honeypot-nids/
│
├── preprocessing/
│   ├── schema.py                # shared constants (column names, label values)
│   └── preprocess.py            # raw CICIDS2017 CSVs → cleaned, scaled splits
│
├── baseline_model/
│   ├── model.py                 # MLP classifier with named blocks for trapdoor hooking
│   ├── dataset.py               # PyTorch Dataset wrapper for CICIDS2017 splits
│   ├── utils.py                 # seed, device selection, metric logger
│   ├── train.py                 # training loop with early stopping
│   ├── evaluate.py              # testset evaluation with full metrics
│   └── result/                  # saved model weights
│
├── data/
│   ├── raw/                     # CICIDS2017 CSVs (not in github)
│   ├── processed/               # scaler, encoders, class weights
│   └── splits/                  # train/val/test CSVs (not in github)
│
└── requirements.txt
```

## Dataset
[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — 2.83M network flow records across 8 capture days, covering benign traffic and 14 attack types (DDoS, PortScan, brute force, web attacks, botnet, infiltration, heartbleed). After cleaning (NaN/Inf removal, deduplication): ~2.52M rows and 78 features.