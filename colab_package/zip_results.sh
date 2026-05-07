#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints results
zip -r asl_training_results.zip checkpoints results
echo "Created asl_training_results.zip"
