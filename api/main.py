"""HTTP API untuk RAG perpajakan.

Satu proses, satu worker: VRAM 6 GB hanya cukup untuk satu inferensi LLM
sekaligus, jadi menambah worker bukan mempercepat melainkan memperebutkan GPU.
"""

import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from taxrag.models import OLLAMA_URL, callbacks
from taxrag.rag import COLLECTION, buat_chain
from taxrag.retrieval import MODE, QDRANT_URL, TOP_N

REFUSAL = "tidak ditemukan dalam dokumen"
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]

# diisi saat startup, dipakai semua request
state: dict = {}


# ------------------------------------------------------------------ skema


class Pertanyaan(BaseModel):
    pertanyaan: str = Field(
        min_length=3, max_length=500, examples=["Biaya jabatan maksimal berapa setahun?"]
    )


class Sitasi(BaseModel):
    doc: str
    pasal: str | None = None


class Potongan(BaseModel):
    doc: str
    pasal: str | None = None
    teks: str


class Jawaban(BaseModel):
    jawaban: str
    ditemukan: bool = Field(
        description="False kalau sistem menolak menjawab karena tidak ada di dokumen"
    )
    sitasi: list[Sitasi]
    potongan: list[Potongan]
    latensi_ms: int


class Kesehatan(BaseModel):
    status: str
    qdrant: str
    llm: str
    jumlah_titik: int | None = None


# ------------------------------------------------------------------ app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # dimuat sekali, bukan per request: bge-m3 butuh beberapa detik
    print("[memuat chain...]")
    state["chain"] = buat_chain()
    print("[siap]")
    yield
    state.clear()


app = FastAPI(
    title="Tax RAG Indonesia",
    description="Tanya-jawab dokumen perpajakan Indonesia, berjalan lokal.",
    version="1.0.0",
    lifespan=lifespan,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )


# ------------------------------------------------------------------ endpoint


@app.get("/")
def akar():
    return {
        "nama": "Tax RAG Indonesia",
        "versi": "1.0.0",
        "mode_retrieval": MODE,
        "top_n": TOP_N,
        "dokumentasi": "/docs",
        "endpoint": "POST /v1/tanya",
    }


@app.get("/health", response_model=Kesehatan)
def health():
    """Benar-benar mengecek Qdrant dan LLM, bukan sekadar membalas 'ok'."""
    qdrant, titik = "mati", None
    try:
        r = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=3)
        if r.status_code == 200:
            titik = r.json()["result"]["points_count"]
            qdrant = "siap" if titik else "kosong"
    except httpx.HTTPError:
        pass

    llm = "mati"
    try:
        if httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200:
            llm = "siap"
    except httpx.HTTPError:
        pass

    sehat = qdrant == "siap" and llm == "siap"
    hasil = Kesehatan(
        status="sehat" if sehat else "tidak sehat",
        qdrant=qdrant,
        llm=llm,
        jumlah_titik=titik,
    )
    if not sehat:
        raise HTTPException(status_code=503, detail=hasil.model_dump())
    return hasil


@app.post("/v1/tanya", response_model=Jawaban)
def tanya(req: Pertanyaan):
    t0 = time.perf_counter()
    try:
        hasil = state["chain"].invoke(
            {"pertanyaan": req.pertanyaan}, config={"callbacks": callbacks()}
        )
    except Exception as e:
        # layanan di belakang mati -> 503, bukan 500 dengan traceback
        raise HTTPException(
            status_code=503,
            detail=f"layanan di belakang tidak bisa dihubungi: {type(e).__name__}",
        ) from e

    jawaban = hasil["jawaban"].strip()
    docs = hasil["docs"]
    ditemukan = REFUSAL not in jawaban.lower()

    return Jawaban(
        jawaban=jawaban,
        ditemukan=ditemukan,
        sitasi=[
            Sitasi(doc=d.metadata.get("doc", "?"), pasal=d.metadata.get("pasal") or None)
            for d in docs
        ]
        if ditemukan
        else [],
        potongan=[
            Potongan(
                doc=d.metadata.get("doc", "?"),
                pasal=d.metadata.get("pasal") or None,
                teks=d.page_content,
            )
            for d in docs
        ],
        latensi_ms=int((time.perf_counter() - t0) * 1000),
    )
