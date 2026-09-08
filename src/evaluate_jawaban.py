"""Ukur kualitas jawaban akhir, bukan cuma retrieval."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from rag import buat_chain
from shared.models import get_tracer

GOLDEN = Path("eval/golden_set.jsonl")
HASIL = Path("eval/hasil_jawaban.json")
REFUSAL = "tidak ditemukan dalam dokumen"


def token_khas(teks: str) -> list[str]:
    """Kata >=5 huruf, angka, atau nominal rupiah."""
    STOP = {"dalam", "untuk", "dengan", "adalah", "sebesar", "paling",
            "tidak", "yang", "atau", "beserta", "sesuai", "karena"}
    kata = re.findall(r"[a-zA-Z]{5,}|[\d.]+%?|rp[\d.]+", teks.lower())
    return [k for k in kata if k not in STOP]


def kunci_utama(benar: str) -> list[str]:
    """Ambil hanya token PALING membedakan: angka, nominal, persen.
    Kata biasa terlalu bervariasi cara pengungkapannya."""
    angka = re.findall(r"rp[\d.]+|[\d.]+%|\b\d+\b", benar.lower())
    if angka:
        return angka
    # kalau jawaban tidak mengandung angka, pakai kata benda khas
    STOP = {"dalam","untuk","dengan","adalah","sebesar","paling","tidak",
            "yang","atau","beserta","sesuai","karena","dapat","harus"}
    return [k for k in re.findall(r"[a-z]{6,}", benar.lower()) if k not in STOP][:4]


def skor_kemiripan(jawaban: str, benar: str) -> float:
    tok = kunci_utama(benar)
    if not tok:
        return 0.0
    j = jawaban.lower().replace(".", "").replace(",", "")
    return sum(1 for t in tok if t.replace(".", "").replace(",", "") in j) / len(tok)


if __name__ == "__main__":
    chain = buat_chain()
    tracer = get_tracer()

    soal = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]

    hasil, waktu = [], []
    for q in tqdm(soal, desc="menjawab"):
        t0 = time.perf_counter()
        out = chain.invoke({"pertanyaan": q["question"]},
                           config={"callbacks": [tracer]})
        waktu.append(time.perf_counter() - t0)

        jawaban = out["jawaban"].strip()
        menolak = REFUSAL in jawaban.lower()

        baris = {
            "id": q["id"],
            "kategori": q["category"],
            "pertanyaan": q["question"],
            "jawaban": jawaban,
            "menolak": menolak,
            "chunk": [f"{d.metadata.get('doc')} {d.metadata.get('pasal') or d.metadata.get('bab') or ''}".strip()
                      for d in out["docs"]],
        }

        if q["category"] == "unanswerable":
            baris["benar"] = menolak          # benar = berhasil menolak
        else:
            baris["skor"] = skor_kemiripan(jawaban, q["answer"])
            baris["benar"] = baris["skor"] >= 0.5 and not menolak

        hasil.append(baris)

    # ---------------------------------------------------------- laporan
    jawab_soal = [h for h in hasil if h["kategori"] != "unanswerable"]
    tolak_soal = [h for h in hasil if h["kategori"] == "unanswerable"]

    akurasi = sum(h["benar"] for h in jawab_soal) / len(jawab_soal)
    refusal = sum(h["benar"] for h in tolak_soal) / len(tolak_soal)
    halusinasi = sum(1 for h in tolak_soal if not h["benar"]) / len(tolak_soal)
    tolak_palsu = sum(1 for h in jawab_soal if h["menolak"]) / len(jawab_soal)

    print()
    print("=" * 58)
    print("KUALITAS JAWABAN")
    print("=" * 58)
    print(f"akurasi jawaban   : {akurasi:.3f}  ({len(jawab_soal)} soal)")
    print(f"refusal rate      : {refusal:.3f}  ({len(tolak_soal)} soal unanswerable)")
    print(f"halusinasi        : {halusinasi:.3f}  <- makin kecil makin baik")
    print(f"menolak padahal bisa: {tolak_palsu:.3f}  <- makin kecil makin baik")
    print(f"latensi rata-rata : {sum(waktu)/len(waktu):.1f} detik")

    print("\nper kategori:")
    for kat in sorted({h["kategori"] for h in hasil}):
        sub = [h for h in hasil if h["kategori"] == kat]
        print(f"  {kat:14s} n={len(sub):2d} | benar {sum(h['benar'] for h in sub)}/{len(sub)}")

    print("\nYANG SALAH:")
    for h in hasil:
        if not h["benar"]:
            print(f"\n  {h['id']} ({h['kategori']})")
            print(f"    tanya : {h['pertanyaan'][:60]}")
            print(f"    jawab : {h['jawaban'][:100]}")
            print(f"    chunk : {h['chunk'][:3]}")

    HASIL.write_text(json.dumps({
        "akurasi": akurasi, "refusal_rate": refusal,
        "halusinasi": halusinasi, "tolak_palsu": tolak_palsu,
        "latensi_rata": sum(waktu) / len(waktu),
        "detail": hasil,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndisimpan -> {HASIL}")