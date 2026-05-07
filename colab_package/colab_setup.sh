#!/usr/bin/env bash
set -euo pipefail

if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi

mkdir -p checkpoints results results/eval results/train_logs results/manual_checks

required_files=(
  "data/active/aslg_pc12_pretrain.json"
  "data/active/project_finetune_v2_v4_contrastive.json"
)

for path in "${required_files[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

echo "Colab setup complete. Required datasets are present."
