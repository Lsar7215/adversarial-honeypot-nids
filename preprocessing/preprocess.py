"""Preprocessing pipeline: raw CICIDS2017 CSVs -> cleaned, scaled, split data.

1. Load all 8 CSV files and combine them
2. Strip whitespace from column names (CICIDS2017 has " Label" and "Label")
3. Drop identifier columns (IPs, timestamps) to prevent data leakage (to the model)
4. Clean: convert to numeric, replace Inf with NaN, drop NaN rows, drop duplicates
5. Create label columns: attack_type, label_binary, label_multiclass
6. Stratified train/val/test split (70/15/15)
7. Fit StandardScaler on train only, apply to all splits (what StandardScaler does is balance the inputs so no single feature dominates the others)
8. Save everything

Run from the project root:
    python -m preprocessing.preprocess
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from preprocessing.schema import (
    LABEL_COLUMN_CANDIDATES,
    BENIGN_VALUE,
    IDENTIFIER_COLUMN_PATTERNS,
    ATTACK_TYPE_COL,
    LABEL_BINARY_COL,
    LABEL_MULTICLASS_COL,
)


# ============================================================
# STEP 1: Load raw CSVs
# ============================================================

def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    # Load all of them and concatenate into one big DataFrame
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Place CSVs there first."
        )

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        frames.append(df)
        print(f"  loaded {path.name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw data: {len(combined):,} rows, {combined.shape[1]} columns")
    return combined


# ============================================================
# STEP 2: Find the label column
# ============================================================

def find_label_column(df: pd.DataFrame) -> str:
    # Auto-detect which column contains the ground-truth labels
    # Build a lowercase→original mapping of all column names
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        f"Could not find a label column. Tried {LABEL_COLUMN_CANDIDATES} "
        f"(case-insensitive) among columns: {list(df.columns)}"
    )


# ============================================================
# STEP 3: Drop identifier columns
# ============================================================
# This is the "anti-cheating" step. Data has IPs, timestamps, flow IDs 
# that the model would happily memorize instead of learning actual traffic patterns

def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove columns that identify specific flows/hosts
    #
    # We compare column names (lowercased) against our blocklist
    # Destination Port is NOT in the blocklist, it's kept because
    # which port is being targeted (80=web, 22=SSH, 443=HTTPS) is
    # genuinely useful for detecting certain attack types
    to_drop = [c for c in df.columns if c.lower() in IDENTIFIER_COLUMN_PATTERNS]
    if to_drop:
        print(f"Dropping identifier columns: {to_drop}")
    return df.drop(columns=to_drop, errors="ignore")


# ============================================================
# STEP 4: Clean features
# ============================================================
# Real data is messy. CICIDS2017 has:
# - Some columns that loaded as strings instead of numbers
# - Inf/-Inf values from divide-by-zero in flow rate calculations
# - Rows with NaN (missing values)
# - Exact duplicate rows

def clean_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    # Convert to numeric, handle Inf/NaN, remove duplicates

    # Everything except the label column should be a number
    feature_cols = [c for c in df.columns if c != label_col]

    # Force to numeric — anything that can't be converted becomes NaN
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace Inf/-Inf with NaN (so dropna catches them too)
    n_before = len(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    n_after_nan = len(df)
    print(
        f"Dropped {n_before - n_after_nan:,} rows with NaN/Inf "
        f"({n_before:,} -> {n_after_nan:,})"
    )

    # Drop exact duplicate rows
    df = df.drop_duplicates()
    n_after_dup = len(df)
    print(
        f"Dropped {n_after_nan - n_after_dup:,} duplicate rows "
        f"({n_after_nan:,} -> {n_after_dup:,})"
    )

    return df.reset_index(drop=True)


# ============================================================
# STEP 5: Derive labels
# ============================================================
# The raw CSV has one label column with strings like "BENIGN",
# "DDoS", "PortScan", etc. We need to create:
# - attack_type: keep the original string (for per-attack analysis)
# - label_binary: 0 = benign, 1 = any attack (for the main task)
# - label_multiclass: integer-encoded attack type (for optional multiclass task)
#

def derive_labels(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, LabelEncoder]:
    # Create binary and multiclass label columns from raw label strings
    # 
    # Returns the modified DataFrame and the fitted LabelEncoder
    # (saved later so evaluate.py can decode integer labels back to strings)

    # Clean up: strip whitespace from label values too
    attack_type = df[label_col].astype(str).str.strip()
    df[ATTACK_TYPE_COL] = attack_type

    # Binary: is it BENIGN (0) or any attack (1)?
    # .str.upper() handles case inconsistencies ("Benign" vs "BENIGN")
    df[LABEL_BINARY_COL] = (attack_type.str.upper() != BENIGN_VALUE).astype(int)

    # Multiclass: encode each attack type as an integer
    # LabelEncoder maps: "BENIGN"→0, "Bot"→1, "DDoS"→2, etc.
    encoder = LabelEncoder()
    df[LABEL_MULTICLASS_COL] = encoder.fit_transform(attack_type)

    # Drop the original label column if it's not one of our standard names
    # (avoids having a redundant column hanging around)
    if label_col not in (ATTACK_TYPE_COL, LABEL_BINARY_COL, LABEL_MULTICLASS_COL):
        df = df.drop(columns=[label_col])

    return df, encoder


# ============================================================
# STEP 6: Report class distribution
# ============================================================
# Before splitting, print how many rows each class has so you can
# 1. Verify the data loaded correctly
# 2. Spot extreme imbalance 

def report_class_distribution(df: pd.DataFrame) -> dict:
    # Print and return class counts for both binary and multiclass
    binary_counts = df[LABEL_BINARY_COL].value_counts().to_dict()
    attack_counts = df[ATTACK_TYPE_COL].value_counts().to_dict()

    report = {
        "binary": {
            ("benign" if k == 0 else "malicious"): v
            for k, v in binary_counts.items()
        },
        "by_attack_type": attack_counts,
        "total_rows": len(df),
    }

    print("Class distribution (binary):", report["binary"])
    print("Class distribution (by attack type):")
    for name, count in sorted(attack_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<30} {count:>10,}")

    return report


# ============================================================
# STEP 7: Stratified split
# ============================================================
# Some attack types (Heartbleed: 11 rows, SQL Injection: 13 rows)
# are so tiny that stratifying on multiclass would fail
# It can't guarantee every class appears in every split with <15 total samples
# Stratifying on binary (benign vs malicious) guarantees at least that
# both major classes are proportionally represented

def stratified_split(df: pd.DataFrame, label_col: str, seed: int):
    # Split into train (70%), val (15%), test (15%).

    # Two steps process:
    # 1. Split off 30% as temp (val + test combined)
    # 2. Split temp 50/50 into val and test
    # Result: 70/15/15

    # Stratify ensures each split has the same benign/malicious ratio
    # Without it, random chance might put mostly benign in one split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[label_col], random_state=seed
    )

    print(
        f"Split sizes -> train: {len(train_df):,}, "
        f"val: {len(val_df):,}, test: {len(test_df):,}"
    )

    return (
        train_df.reset_index(drop=True).copy(),
        val_df.reset_index(drop=True).copy(),
        test_df.reset_index(drop=True).copy(),
    )


# ============================================================
# STEP 8: Compute class weights
# ============================================================
# Compute class weifht properly from the actual class frequencies
#
# sklearn's compute_class_weight("balanced") does:
#   weight_for_class_i = total_samples / (num_classes * count_of_class_i)
#
# With 2.09M benign and 426K malicious:
#   benign weight  ≈ 2.52M / (2 * 2.09M) ≈ 0.60
#   malicious weight ≈ 2.52M / (2 * 426K) ≈ 2.96
#
# This means mistakes on malicious traffic are penalized  around 5 more times
# than mistakes on benign, forcing the model to actually learn to
# detect attacks rather than lazily predicting "benign" always

def compute_and_save_class_weights(train_df: pd.DataFrame, out_dir: Path) -> None:
    # Compute balanced class weights for both binary and multiclass tasks
    for label_col, name in [
        (LABEL_BINARY_COL, "binary"),
        (LABEL_MULTICLASS_COL, "multiclass"),
    ]:
        classes = np.sort(train_df[label_col].unique())
        weights = compute_class_weight(
            "balanced", classes=classes, y=train_df[label_col].values
        )
        mapping = {int(c): float(w) for c, w in zip(classes, weights)}

        with open(out_dir / f"class_weights_{name}.json", "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved class_weights_{name}.json: {mapping}")


# ============================================================
# MAIN: Run the full pipeline
# ============================================================

def main():
    raw_dir = Path("data/raw")
    out_processed = Path("data/processed")
    out_splits = Path("data/splits")
    seed = 42

    out_processed.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    print("=== Loading raw CSVs ===")
    df = load_raw_csvs(raw_dir)

    # Step 2: Find label column
    label_col = find_label_column(df)
    print(f"Detected label column: '{label_col}'")

    # Step 3: Drop identifiers
    df = drop_identifier_columns(df)

    # Step 4: Clean
    print("=== Cleaning features ===")
    df = clean_features(df, label_col)

    # Step 5: Derive labels
    print("=== Deriving labels ===")
    df, label_encoder = derive_labels(df, label_col)

    # Step 6: Report distribution
    print("=== Class distribution ===")
    distribution = report_class_distribution(df)
    with open(out_processed / "class_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)

    # Step 7: Split
    print("=== Splitting ===")
    train_df, val_df, test_df = stratified_split(df, LABEL_BINARY_COL, seed)

    # Identify which columns are features (everything that's not a label)
    feature_cols = [
        c for c in df.columns
        if c not in (ATTACK_TYPE_COL, LABEL_BINARY_COL, LABEL_MULTICLASS_COL)
    ]
    print(f"Feature count: {len(feature_cols)}")

    # Step 8: Scale
    # Fit on train only, then transform all three splits.
    print("=== Scaling (fit on train only) ===")
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Step 9: Save everything
    print("=== Saving artifacts ===")
    joblib.dump(scaler, out_processed / "scaler.joblib")
    joblib.dump(label_encoder, out_processed / "label_encoder.joblib")
    with open(out_processed / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    compute_and_save_class_weights(train_df, out_processed)

    train_df.to_csv(out_splits / "train.csv", index=False)
    val_df.to_csv(out_splits / "val.csv", index=False)
    test_df.to_csv(out_splits / "test.csv", index=False)
    print(f"Saved splits to {out_splits}")

    print("Done.")


if __name__ == "__main__":
    main()