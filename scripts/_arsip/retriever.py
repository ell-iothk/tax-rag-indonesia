"""Bangun retriever: dense, sparse, atau hybrid."""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore

try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from shared.models import get_embeddings, get_reranker

CHUNKS = Path("data/processed/chunks.jsonl")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "tax_docs"


def muat_docs() -> list[Document]:
    docs = []
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        teks = c.pop("text")
        docs.append(Document(page_content=teks, metadata=c))
    return docs


def dense_retriever(k: int = 10):
    store = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )
    return store.as_retriever(search_kwargs={"k": k})


def sparse_retriever(k: int = 10):
    r = BM25Retriever.from_documents(muat_docs())
    r.k = k
    return r


def hybrid_retriever(k: int = 10, bobot: tuple[float, float] = (0.5, 0.5),
                     k_kandidat: int | None = None):
    """k_kandidat = berapa yang diambil tiap retriever sebelum difusi."""
    kk = k_kandidat or k * 3
    return EnsembleRetriever(
        retrievers=[dense_retriever(kk), sparse_retriever(kk)],
        weights=list(bobot),
    )

def rerank(base_retriever, top_n: int = 5):
    """Bungkus retriever dengan cross-encoder reranker."""
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def dense_rerank_retriever(top_n: int = 5, k_kandidat: int = 20):
    """Pola dua tahap: bi-encoder ambil banyak, cross-encoder pilih terbaik."""
    return rerank(dense_retriever(k_kandidat), top_n=top_n)