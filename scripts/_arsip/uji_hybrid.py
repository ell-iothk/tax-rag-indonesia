"""Bandingkan dense vs sparse vs hybrid pada soal yang gagal."""
from shared.retrieval import dense_retriever, sparse_retriever, hybrid_retriever

SOAL = {
    "q028 (lexical)": "Apa isi Pasal 21 ayat (1) PMK 168 Tahun 2023?",
    "q016 (multi_hop)": "Saya lajang tanpa tanggungan, gaji 10 juta sebulan, kena TER berapa persen?",
}

retrievers = {
    "dense ": dense_retriever(k=5),
    "sparse": sparse_retriever(k=5),
    "hybrid": hybrid_retriever(k=5),
}

for nama_soal, q in SOAL.items():
    print("=" * 70)
    print(nama_soal, "|", q)
    for nama_r, r in retrievers.items():
        hasil = r.invoke(q)
        label = [
            (d.metadata.get("pasal") or d.metadata.get("bab") or "?")[:22]
            for d in hasil[:5]
        ]
        print(f"  {nama_r}: {label}")
    print()