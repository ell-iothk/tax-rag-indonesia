"""Entrypoint container: siapkan prasyarat lalu jalankan API.

Dibuat idempoten supaya `docker compose up` kedua tidak mengulang pekerjaan:
model hanya diunduh kalau folder kosong, index hanya dibangun kalau koleksi
Qdrant belum ada titiknya.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from taxrag.models import MODELS
from taxrag.rag import COLLECTION
from taxrag.retrieval import QDRANT_URL

ROOT = Path(__file__).resolve().parent.parent
BATAS_TUNGGU = int(os.getenv("BATAS_TUNGGU_QDRANT", "120"))


def jalankan(*perintah: str) -> None:
    print(f"[entrypoint] {' '.join(perintah)}", flush=True)
    subprocess.run([sys.executable, *perintah], cwd=ROOT, check=True)


def tunggu_qdrant() -> None:
    for _ in range(BATAS_TUNGGU):
        try:
            if httpx.get(f"{QDRANT_URL}/readyz", timeout=2).status_code == 200:
                print("[entrypoint] qdrant siap", flush=True)
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    sys.exit(f"[entrypoint] qdrant tidak siap setelah {BATAS_TUNGGU}s: {QDRANT_URL}")


def jumlah_titik() -> int:
    r = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    return r.json()["result"]["points_count"] or 0


def siapkan_model() -> None:
    if (MODELS / "bge-m3").is_dir():
        print("[entrypoint] model sudah ada", flush=True)
        return
    print("[entrypoint] mengunduh model, sekitar 4,4 GB, sekali saja", flush=True)
    jalankan("scripts/download_models.py")


def siapkan_index() -> None:
    if jumlah_titik():
        print(f"[entrypoint] index sudah ada: {jumlah_titik()} titik", flush=True)
        return
    print("[entrypoint] index kosong, membangun dari PDF", flush=True)
    jalankan("scripts/ingest.py")
    jalankan("scripts/index_qdrant.py")


if __name__ == "__main__":
    siapkan_model()
    tunggu_qdrant()
    siapkan_index()

    print("[entrypoint] menjalankan uvicorn", flush=True)
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--workers",
            "1",
        ],
    )
