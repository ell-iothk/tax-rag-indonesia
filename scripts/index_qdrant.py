"""Muat chunk ke Qdrant."""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from taxrag.models import get_embeddings

CHUNKS = Path("data/processed/chunks.jsonl")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "tax_docs"


def muat_chunks() -> list[Document]:
    docs = []
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        teks = c.pop("text")
        docs.append(Document(page_content=teks, metadata=c))
    return docs


if __name__ == "__main__":
    docs = muat_chunks()
    print(f"chunk dimuat: {len(docs)}")

    # hapus collection lama supaya bisa diulang tanpa duplikat
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"collection lama '{COLLECTION}' dihapus")

    emb = get_embeddings()

    print("meng-embed dan menyimpan... (beberapa menit di CPU)")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=emb,
        url=QDRANT_URL,
        collection_name=COLLECTION,
        batch_size=16,
    )

    info = client.get_collection(COLLECTION)
    print()
    print(f"collection  : {COLLECTION}")
    print(f"jumlah titik: {info.points_count}")
