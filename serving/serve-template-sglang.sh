#!/usr/bin/env bash
# Generic SGLang serve script template
# Customize: MODEL_PATH, TP_SIZE, PORT, QUANTIZATION

#SBATCH --job-name=sglang-serve
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --qos=1d

module load cuda/12.9.0

MODEL_PATH="/shared/models/glm-5.1-nvfp4"  # <-- EDIT ME
TP_SIZE=8                                    # <-- EDIT ME (must divide num_heads)
PORT=8103
QUANT="modelopt_fp4"                        # or "fp8", "none"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp "$TP_SIZE" \
  --port "$PORT" \
  --quantization "$QUANT" \
  --mem-fraction-static 0.80 \
  --max-model-len 202000 \
  --enable-mixed-chunk \
  --chunked-prefill-size 131072 \
  --enable-eagle \
  --eagle-model-path "/shared/models/eagle-drafter" \
  --tool-call-parser inline \
  --reasoning-parser inline \
  --trust-remote-code
