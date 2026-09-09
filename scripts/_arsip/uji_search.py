"""Uji pencarian di Qdrant."""
from langchain_qdrant import QdrantVectorStore
from shared.models import get_embeddings

store = QdrantVectorStore.from_existing_collection(
    embedding=get_embeddings(),
    url="http://localhost:6333",
    collection_name="tax_docs",
)

PERTANYAAN = [
    "Biaya jabatan maksimal berapa setahun?",
    "Selebgram masuk kategori apa?",
    "Saya lajang gaji 10 juta, kena tarif berapa?",
    "Resep rendang padang",
]

for q in PERTANYAAN:
    print(f"\n{'='*70}")
    print("TANYA:", q)
    hasil = store.similarity_search_with_score(q, k=3)
    for i, (doc, skor) in enumerate(hasil, 1):
        m = doc.metadata
        label = m.get("pasal") or m.get("bab") or "-"
        isi = doc.page_content.replace("\n", " ")
        print(f"\n  [{i}] skor {skor:.4f} | {m['doc']} | {label[:40]}")
        print(f"      {isi[:150]}")