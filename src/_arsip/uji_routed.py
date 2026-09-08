from shared.retrieval import get_retriever, klasifikasi_query

SOAL = {
    "q028": "Apa isi Pasal 21 ayat (1) PMK 168 Tahun 2023?",
    "q029": "Lampiran huruf D PP 58 Tahun 2023 berisi apa?",
    "q030": "PP 58 Tahun 2023 mencabut ketentuan apa?",
    "q016": "Saya lajang tanpa tanggungan, gaji 10 juta sebulan, kena TER berapa persen?",
    "q001": "Biaya jabatan itu berapa persen dan maksimal berapa?",
}

hybrid = get_retriever("tax_docs", "data/processed/chunks.jsonl", mode="hybrid")
routed = get_retriever("tax_docs", "data/processed/chunks.jsonl", mode="routed")

for nama, q in SOAL.items():
    print("=" * 72)
    print(f"{nama} [{klasifikasi_query(q)}] {q[:52]}")
    for label, r in [("hybrid", hybrid), ("routed", routed)]:
        h = r.invoke(q)
        isi = [(d.metadata.get("pasal") or d.metadata.get("bab") or "?")[:20]
               for d in h]
        print(f"  {label}: {isi}")
    print()