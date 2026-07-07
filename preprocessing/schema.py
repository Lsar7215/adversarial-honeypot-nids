"""Column-name and label conventions for the CICIDS2017 dataset.

FILE PURPOSE:
Standardized varies raw data source column names into a single unified format.
Ensures naming consistency across preprocessing, training, and evaluation pipelines.
"""

from __future__ import annotations

# --- Label column detection ---
# CICIDS2017 CSVs might call the label column different things
LABEL_COLUMN_CANDIDATES = ["label", "class", "attack_type"]

# The exact string CICIDS2017 uses for non-malicious traffic
BENIGN_VALUE = "BENIGN"

# --- Columns to DROP ---
# Standard identifier columns dropped in CICIDS2017 ML-NIDS research.
# These describe WHO/WHEN/WHERE, not WHAT the traffic does.
# Keeping them would let the model memorize the lab topology
# (specific attacker IPs, scheduled attack times) instead of
# learning generalizable traffic patterns.
#
# Destination port is correlates with which service is being targeted 
# (e.g. port 80 = web, port 22 = SSH), which is really useful for detecting certain attacks.
IDENTIFIER_COLUMN_PATTERNS = [
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "timestamp",
    "source port",
    "src port",
]

# --- Output column names ---
# Every file downstream (dataset.py, train.py, evaluate.py) imports these
# so there's zero chance of a typo mismatch.
ATTACK_TYPE_COL = "attack_type"           # original string label: "BENIGN", "DDoS", etc.
LABEL_BINARY_COL = "label_binary"         # 0 = benign, 1 = malicious
LABEL_MULTICLASS_COL = "label_multiclass" # integer-encoded: 0, 1, 2, ... 14
# integer-encoded attack types (15 classes, alphabetical):
# 0=BENIGN, 1=Bot, 2=DDoS, 3=DoS GoldenEye, 4=DoS Hulk,
# 5=DoS Slowhttptest, 6=DoS slowloris, 7=FTP-Patator,
# 8=Heartbleed, 9=Infiltration, 10=PortScan, 11=SSH-Patator,
# 12=Web Attack Brute Force, 13=Web Attack SQL Injection,
# 14=Web Attack XSS

# To test the file, run:
#   python -c "from preprocessing.schema import *; print(IDENTIFIER_COLUMN_PATTERNS)"
# It should print a list of strings like ["flow id", "source ip", ...]