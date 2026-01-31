#!/bin/bash
# Quick evaluation script

# Change to script directory
cd "$(dirname "$0")"

# Default model path
MODEL="${1:-models/runs/training_runs/yolo11n_s2_10m_4b_v1/weights/best.pt}"

echo "Evaluating model: $MODEL"
python scripts/10_evaluate_model.py --model "$MODEL"
