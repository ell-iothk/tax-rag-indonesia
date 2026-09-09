"""Lihat konteks persis yang diterima LLM."""
from shared.retrieval import get_retriever

r = get_retriever("tax_docs", "data/processed/chunks.jsonl")
Q = "Siapa saja yang dianggap tenaga ahli?"

for i, d in enumerate(r.invoke(Q), 1):
    m = d.metadata
    ada = "PENGACARA ADA" if "pengacara" in d.page_content.lower() else ""
    print(f"--- [{i}] {m.get('pasal') or m.get('bab')} {ada}")
    print(d.page_content[:420].replace("\n", " "))
    print()