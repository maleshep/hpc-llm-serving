# OpenWispr

OpenWispr is the self-hosted speech stack in this repo. The current default path is a single FastAPI server on port `8200` that loads model backends from [`model_registry.yaml`](model_registry.yaml).

## Current architecture

- Server: [`server.py`](server.py)
- Active ASR backend: `faster-whisper`
- Active TTS backend: `f5-tts`
- Config: [`model_registry.yaml`](model_registry.yaml)
- Windows client: [`client/open_wispr.pyw`](client/open_wispr.pyw)
- Web UI: [`web/`](web)

Legacy backends are still available for A/B testing:

- `qwen-audio` ASR in [`backends/legacy_qwen.py`](backends/legacy_qwen.py)
- `cosyvoice` TTS in [`backends/legacy_cosyvoice.py`](backends/legacy_cosyvoice.py)

## Ports

- `8200`: unified OpenWispr API
- `8280`: optional local web UI if you run `python open_wispr/web/server.py --port 8280`

The Windows hotkey client only needs `8200`.

## Quick start

### 1. Start the speech server on HPC

```bash
sbatch open_wispr/serve-unified.sh
```

### 2. Forward the speech port locally

```bash
ssh -L 8200:NODE:8200 -N user@hpc-cluster.example.com
```

### 3. Run the Windows client

```bash
python open_wispr/client/open_wispr.pyw
```

## API

### `GET /health`

Returns aggregate backend health:

```json
{
  "status": "ok",
  "asr": { "backend": "faster-whisper", "status": "ok" },
  "tts": { "backend": "f5-tts", "status": "ok" }
}
```

### `POST /transcribe`

Form fields:

- `audio`: audio file
- `language`: optional, default `auto`

Response:

```json
{
  "text": "transcribed text",
  "language": "en",
  "processing_ms": 241.7
}
```

### `POST /synthesize`

Form fields:

- `text`
- `voice` optional, default `default`
- `speed` optional, default `1.0`

Response is `audio/wav`.

### `POST /clone`

Form fields:

- `reference`
- `text`

Response is `audio/wav`.

### `POST /admin/swap-model`

Hot-swap the active backend without restarting the process.

Form fields:

- `service`: `asr` or `tts`
- `backend_name`

## Deployment notes

- [`serve-unified.sh`](serve-unified.sh) is the current Slurm entry point for the unified speech server.
- [`serve-v2.sh`](serve-v2.sh) is an older experimental combined stack that also launched a colocated LLM.
- [`asr_server.py`](asr_server.py) and [`tts_server.py`](tts_server.py) are legacy split-server implementations kept for reference and fallback work.
