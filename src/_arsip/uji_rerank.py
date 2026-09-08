"""Bandingkan dense biasa vs dense+rerank pada soal yang gagal."""
from shared.retrieval import dense_retriever, dense_rerank_retriever

SOAL = {
    "q028 cari 'Pasal 21'":
        "Apa isi Pasal 21 ayat (1) PMK 168 Tahun 2023?",
    "q016 cari 'Lampiran huruf A'":
        "Saya lajang tanpa tanggungan, gaji 10 juta sebulan, kena TER berapa persen?",
    "q001 cari 'Pasal 10'":
        "Biaya jabatan itu berapa persen dan maksimal berapa?",
}

polos = dense_retriever(k=5)
rerank = dense_rerank_retriever(top_n=5, k_kandidat=20)

for nama, q in SOAL.items():
    print("=" * 68)
    print(nama)
    for label, r in [("dense ", polos), ("rerank", rerank)]:
        hasil = r.invoke(q)[:5]
        isi = [(d.metadata.get("pasal") or d.metadata.get("bab") or "?")[:20]
               for d in hasil]
        print(f"  {label}: {isi}")
    print()