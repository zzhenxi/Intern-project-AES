#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
cd "$PROJECT_ROOT/src"

API_KEY=""  # 여기에 직접 API 키 입력하거나 실행할 때 입력
DATASET_NAME="asap"
DATASET_PATH="../datasets/asap_essay_set_1.jsonl"
N_SAMPLES=50
MODEL_NAME="gpt-4.1-mini"
N_AGENTS=4
RESULTS_DIR="../results"

python main.py \
  --api_key "$API_KEY" \
  --dataset_name "$DATASET_NAME" \
  --dataset_path "$DATASET_PATH" \
  --n_samples $N_SAMPLES \
  --model_name "$MODEL_NAME" \
  --n_agents $N_AGENTS \
  --results_dir "$RESULTS_DIR"

# "--feedback"
