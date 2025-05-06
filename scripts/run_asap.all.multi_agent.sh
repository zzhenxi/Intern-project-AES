#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
cd "$PROJECT_ROOT/src"

API_KEY=""  # 여기에 직접 API 키 입력하거나 실행할 때 입력
DATASET_NAME="asap"
MODEL_NAME="gpt-4.1-mini"
N_SAMPLES=50
N_AGENTS=4
RESULTS_DIR="../results"

for i in {1..8}; do
  DATASET_PATH="../datasets/asap_essay_set_${i}.jsonl"
  echo "Running evaluation for: $DATASET_PATH"
  
  python main.py \
    --api_key "$API_KEY" \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$DATASET_PATH" \
    --n_samples $N_SAMPLES \
    --model_name "$MODEL_NAME" \
    --n_agents $N_AGENTS \
    --results_dir "$RESULTS_DIR"\
    --multi_agent
done