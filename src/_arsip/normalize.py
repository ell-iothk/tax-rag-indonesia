"""Bersihkan teks hasil parsing dan tandai kualitasnya."""
import json
import re
from pathlib import Path

PROC = Path("data/processed")

# Penanda halaman tabel: spasi antar-kolom harus dipertahankan
PENANDA_TABEL = ["Penghasilan Bruto Bulanan", "Penghasilan Bruto Harian"]

# Pola kerusakan OCR
POLA_RUSAK = [
    r"\d[OolI]",        # digit diikuti huruf mirip: 5.4O
    r"[OolI]\d",        # huruf mirip diikuti digit: O0
    r"o/o",             # persen rusak, tanpa \b
    r"[A-Z]{15,}",      # spasi hilang
    r"[A-Z][.,][A-Z]",  # titik/koma nyasar: PENGHASII.,AN
]


def halaman_tabel(teks: str) -> bool:
    # judul kolom (halaman pertama tabel)
    if any(p in teks for p in PENANDA_TABEL):
        return True
    # halaman lanjutan: banyak baris berpola "di atas Rp... " + "persen)"
    n_rentang = teks.count("di atas Rp")
    n_persen = teks.count("persen)")
    return n_rentang >= 3 and n_persen >= 3


def skor_kerusakan(teks: str) -> int:
    """Hitung total kemunculan pola rusak."""
    return sum(len(re.findall(p, teks)) for p in POLA_RUSAK)


def bersihkan(teks: str, is_tabel: bool) -> str:
    baris = [b for b in teks.split("\n") if b.strip()]
    if is_tabel:
        # hanya buang spasi di ujung, jangan sentuh spasi antar-kolom
        return "\n".join(b.rstrip() for b in baris)
    # teks biasa: rapikan spasi ganda
    return "\n".join(re.sub(r"\s+", " ", b).strip() for b in baris)


if __name__ == "__main__":
    for src in sorted(PROC.glob("*_pages.jsonl")):
        hasil = []
        for line in open(src, encoding="utf-8"):
            p = json.loads(line)
            is_tabel = halaman_tabel(p["text"])
            skor = skor_kerusakan(p["text"])
            hasil.append({
                "doc": p["doc"],
                "page": p["page"],
                "text": bersihkan(p["text"], is_tabel),
                "is_tabel": is_tabel,
                "skor_rusak": skor,
                "kualitas": "rusak" if skor / max(len(p["text"]), 1) * 1000 > 3 else "bersih",
            })

        out = PROC / src.name.replace("_pages", "_clean")
        with open(out, "w", encoding="utf-8") as f:
            for h in hasil:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")

        n_tabel = sum(h["is_tabel"] for h in hasil)
        n_rusak = sum(h["kualitas"] == "rusak" for h in hasil)
        hemat = 1 - sum(len(h["text"]) for h in hasil) / sum(
            len(json.loads(l)["text"]) for l in open(src, encoding="utf-8"))
        print(f"{src.name}: {len(hasil)} hal | {n_tabel} tabel | "
              f"{n_rusak} rusak | teks berkurang {hemat:.0%}")
