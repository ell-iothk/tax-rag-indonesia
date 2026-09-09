"""Lihat token apa yang dipakai BM25 dan chunk mana yang menang."""
from shared.retrieval import sparse_retriever

r = sparse_retriever("data/processed/chunks.jsonl", k=5)

for q in ["Lampiran huruf D PP 58 Tahun 2023 berisi apa?",
          "Lampiran huruf D tarif efektif harian",
          "tarif efektif harian"]:
    print("=" * 70)
    print("QUERY:", q)
    for d in r.invoke(q):
        m = d.metadata
        print(f"  {m.get('pasal') or m.get('bab') or '(kosong)':45s} | {d.page_content[:55]}")
    print()