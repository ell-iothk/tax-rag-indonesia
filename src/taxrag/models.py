"""Model bersama untuk semua project. Dimuat sekali per proses."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# akar repo = tiga tingkat di atas file ini (src/taxrag/models.py -> akar)
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
load_dotenv(ROOT / ".env")

# jangan unduh format .bin yang lama dan rawan
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")

EMBED_MODEL = str(MODELS / "bge-m3")
RERANK_MODEL = str(MODELS / "bge-reranker-v2-m3")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "qwen3-id"


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"[memuat embedding: {EMBED_MODEL}]")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0):
    from langchain_ollama import ChatOllama

    print(f"[menyambung LLM: {LLM_MODEL}]")
    return ChatOllama(model=LLM_MODEL, temperature=temperature, base_url=OLLAMA_URL)


@lru_cache(maxsize=1)
def get_reranker():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    print(f"[memuat reranker: {RERANK_MODEL}]")
    return HuggingFaceCrossEncoder(
        model_name=RERANK_MODEL,
        model_kwargs={"device": "cpu"},
    )


@lru_cache(maxsize=1)
def get_tracer():
    """Callback handler Langfuse, atau None kalau tracing tidak dikonfigurasi.

    Tracing itu opsional: tanpa key di .env sistem tetap jalan tanpa
    membanjiri terminal dengan error 401.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def callbacks() -> list:
    """Daftar callback untuk chain.invoke(). Kosong kalau tracing mati."""
    tracer = get_tracer()
    return [tracer] if tracer else []
