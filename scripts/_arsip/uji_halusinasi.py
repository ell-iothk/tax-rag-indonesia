"""Uji apakah halusinasi di top_n=4 reproducible."""
import json
from rag import buat_chain

REFUSAL = "tidak ditemukan dalam dokumen"

soal = [json.loads(l) for l in open("eval/golden_set.jsonl", encoding="utf-8")]
soal = [q for q in soal if q["category"] == "unanswerable"]

for n in [4, 5]:
    chain = buat_chain(top_n=n)
    print(f"--- top_n={n} ---")
    for ulang in range(2):
        gagal = []
        for q in soal:
            jw = chain.invoke({"pertanyaan": q["question"]})["jawaban"]
            if REFUSAL not in jw.lower():
                gagal.append((q["id"], jw.strip()[:60]))
        print(f"  jalan {ulang+1}: halusinasi {len(gagal)}/5")
        for i, j in gagal:
            print(f"      {i}: {j}")
    print()