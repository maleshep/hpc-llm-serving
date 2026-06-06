#!/usr/bin/env bash
# Native Transformers serve script (for smaller models / fine-tuned adapters)
# Customize: MODEL_PATH, PORT

#SBATCH --job-name=native-serve
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --qos=1d

module load cuda/12.9.0

MODEL_PATH="/shared/models/gemma-4-31b-mmm-sft"  # <-- EDIT ME
PORT=8200

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 2 \
  --port "$PORT" \
  --max-model-len 8192 \
  --enforce-eager true
