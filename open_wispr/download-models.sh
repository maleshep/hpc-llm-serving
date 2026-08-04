#!/bin/bash
# Download ASR + TTS models to HPC project storage
# Run this once on the login node (I/O-bound, safe)

set -e

PROJECT=/shared/project/hpc-llm
MODELS_DIR=$PROJECT/models

module load cli-tools

mkdir -p $MODELS_DIR

echo "=== Downloading Qwen2-Audio-7B-Instruct (ASR) ==="
echo "Size: ~14 GB"
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct \
    --local-dir $MODELS_DIR/qwen2-audio-7b \
    --local-dir-use-symlinks False

echo ""
echo "=== Downloading CosyVoice3-0.5B (TTS) ==="
echo "Size: ~2 GB"
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B \
    --local-dir $MODELS_DIR/cosyvoice3-0.5b \
    --local-dir-use-symlinks False

echo ""
echo "=== Cloning CosyVoice repo (for inference code) ==="
cd $PROJECT
if [ ! -d "CosyVoice" ]; then
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    pip install -r requirements.txt
fi

echo ""
echo "=== Done ==="
echo "Models at: $MODELS_DIR/"
ls -lh $MODELS_DIR/
