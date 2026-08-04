"""
OpenWispr Web UI server.

Serves the static HTML client for the unified OpenWispr API.
No GPU needed.

Usage:
    python server.py [--port 8280]
"""

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="OpenWispr Web")

# CORS: allow browser on localhost to call ASR/TTS directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_URL = "http://localhost:8200"

WEB_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def root():
    return (WEB_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    """Return health from the unified OpenWispr API."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{API_URL}/health")
            return r.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# Mount static files (for any future CSS/JS assets)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
