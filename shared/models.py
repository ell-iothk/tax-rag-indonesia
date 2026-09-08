"""Model bersama untuk semua project. Dimuat sekali per proses."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# baca .env dari root ai-projects
# akar repo = dua tingkat di atas file ini (shared/models.py -> shared -> akar)
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

# cache model. bisa ditimpa lewat variabel lingkungan HF_HOME
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
# jangan unduh format .bin yang lama dan rawan
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
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
    return ChatOllama(model=LLM_MODEL, temperature=temperature)


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
    """Callback handler Langfuse. Pasang ke chain, semua langkah terekam.

    Dipakai: chain.invoke(x, config={"callbacks": [get_tracer()]})
    """
    from langfuse.langchain import CallbackHandler
    print("[tracing: Langfuse aktif]")
    return CallbackHandler()