"""Cari bobot dense:sparse terbaik pada soal yang gagal."""
from shared.retrieval import hybrid_retriever

SOAL = {
    "q028 cari 'Pasal 21'": (
        "Apa isi Pasal 21 ayat (1) PMK 168 Tahun 2023?", "Pasal 21"),
    "q016 cari 'Lampiran huruf A'": (
        "Saya lajang tanpa tanggungan, gaji 10 juta sebulan, kena TER berapa persen?",
        "Lampiran huruf A"),
}

BOBOT = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]

for nama, (q, target) in SOAL.items():
    print("=" * 66)
    print(nama)
    for w in BOBOT:
        for kk in (5, 10, 20):
            r = hybrid_retriever(k=5, bobot=w, k_kandidat=kk)
            hasil = r.invoke(q)[:5]
            posisi = None
            for i, d in enumerate(hasil, 1):
                label = (d.metadata.get("pasal") or "") + (d.metadata.get("bab") or "")
                if target in label:
                    posisi = i
                    break
            tanda = f"posisi {posisi}" if posisi else "TIDAK ADA"
            print(f"  dense {w[0]:.1f} / sparse {w[1]:.1f} | kandidat {kk:2d} -> {tanda}")
    print()