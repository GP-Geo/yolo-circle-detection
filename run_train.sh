#!/usr/bin/env bash
# Quick training script for Apple Silicon (M1/M2/M3)
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

echo "Starting YOLO training with MPS fallback enabled..."
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/04_train.py
