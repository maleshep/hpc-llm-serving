#!/bin/bash
#SBATCH --account=hpc-llm
#SBATCH --job-name=dl-audio8-tts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --qos=3h
#SBATCH --output=/shared/project/hpc-llm/llm/logs/download-audio8-tts_%j.out

# Download Audio8-TTS-Preview-0.6b from HuggingFace Hub
#
# Model: Audio8/Audio8-TTS-Preview-0.6b
# Size: ~1.2 GB BF16 (601M main model + bundled neural audio codec)
# License: Apache 2.0
#
# What this is:
#   DualAR (Dual Autoregressive) TTS model — Slow AR (24 layers, 896 width,
#   14 attn heads, 2 KV heads) predicts one semantic token per audio frame;
#   Fast AR (4 layers, same width/heads) predicts codec codebooks conditioned
#   on Slow hidden state. Bundled 44.1 kHz neural audio codec (no separate
#   codec checkpoint needed). 11 languages (EN/DE/FR/IT/ES/JP/KR/ZH + NL/PL/CA).
#   Zero-shot voice cloning from a single reference clip. ~1 GiB ONNX INT4
#   variant exists for CPU-only deployment but this script grabs the BF16
#   original for GPU serving on SGLang Omni.
#
# Why we picked it (2026-08-04):
#   Beats Fish S2 Pro (4.6B), Higgs Audio v2 (4.7B), CosyVoice3 (1.5B),
#   MOSS-TTS (8.5B), VoxCPM2 (2.3B) on English WER (1.506, best in class)
#   and on CV3 zh/en multilingual error rates — despite being the smallest
#   model in the comparison. Apache 2.0 license (no community-license drag
#   like MiniMax). SGLang Omni adapter provides OpenAI-compatible
#   /v1/audio/speech endpoint out of the box.
#
# Usage:
#   sbatch serving/download-audio8-tts.sh

set -euo pipefail

PROJECT=/shared/project/hpc-llm
LLM_DIR=$PROJECT/llm
OUTPUT_DIR=$LLM_DIR/models/audio8-tts-0.6b

module purge
module load cuda/12.9.0

source $LLM_DIR/venv/bin/activate

echo "============================================================"
echo "Downloading: Audio8/Audio8-TTS-Preview-0.6b"
echo "Destination: $OUTPUT_DIR"
echo "Expected size: ~1.5 GB (model + bundled codec + tokenizer)"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Audio8/Audio8-TTS-Preview-0.6b',
    local_dir='$OUTPUT_DIR',
    ignore_patterns=['*.gguf', '*.onnx', '*onnx-int4*'],
)
print('Download complete!')
"

echo ""
echo "Download complete: $OUTPUT_DIR"
echo "Size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "Next steps:"
echo "  sbatch serving/serve-audio8-tts.sh    # boot SGLang Omni on 1x L40S, port 8310"
