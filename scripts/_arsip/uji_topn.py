"""Cari top_n terbaik. Trade-off: presisi vs peluang jawaban ada di konteks."""
import json
import re
from pathlib import Path

from tqdm import tqdm

from rag import buat_chain

REFUSAL = "tidak ditemukan dalam dokumen"


def kunci_utama(benar: str):
    angka = re.findall(r"rp[\d.]+|[\d.]+%|\b\d+\b", benar.lower())
    if angka:
        return angka
    STOP = {"dalam","untuk","dengan","adalah","sebesar","paling","tidak",
            "yang","atau","beserta","sesuai","karena","dapat","harus"}
    return [k for k in re.findall(r"[a-z]{6,}", benar.lower())
            if k not in STOP][:4]


def skor(jawaban: str, benar: str) -> float:
    tok = kunci_utama(benar)
    if not tok:
        return 0.0
    j = jawaban.lower().replace(".", "").replace(",", "")
    return sum(1 for t in tok if t.replace(".", "").replace(",", "") in j) / len(tok)


soal = [json.loads(l) for l in open("eval/golden_set.jsonl", encoding="utf-8")]

print(f"{'top_n':6s} {'akurasi':>9s} {'refusal':>9s} {'halusinasi':>11s} {'tolak palsu':>12s}")
print("-" * 52)

for n in [2, 3, 4, 5, 7]:
    chain = buat_chain(top_n=n)
    jawab_ok, tolak_ok, halu, tolak_palsu = 0, 0, 0, 0
    n_jawab = n_tolak = 0

    for q in tqdm(soal, desc=f"top_n={n}", leave=False):
        out = chain.invoke({"pertanyaan": q["question"]})
        jw = out["jawaban"].strip()
        menolak = REFUSAL in jw.lower()

        if q["category"] == "unanswerable":
            n_tolak += 1
            if menolak:
                tolak_ok += 1
            else:
                halu += 1
        else:
            n_jawab += 1
            if skor(jw, q["answer"]) >= 0.5 and not menolak:
                jawab_ok += 1
            if menolak:
                tolak_palsu += 1

    print(f"{n:<6d} {jawab_ok/n_jawab:9.3f} {tolak_ok/n_tolak:9.3f} "
          f"{halu/n_tolak:11.3f} {tolak_palsu/n_jawab:12.3f}")