#!/usr/bin/env bash
# vLLM serve script template
# Customize: MODEL_PATH, TP_SIZE, PORT, ENFORCE_EAGER

#SBATCH --job-name=vllm-serve
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --qos=1d

module load cuda/12.9.0

MODEL_PATH="/shared/models/kimi-k2.6-nvfp4"  # <-- EDIT ME
TP_SIZE=4                                     # <-- EDIT ME
PORT=8104
ENFORCE_EAGER=true                            # set false if torch.compile stable

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --max-model-len 196608 \
  --quantization modelopt_fp4 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi \
  --reasoning-parser kimi \
  --enforce-eager "$ENFORCE_EAGER"
