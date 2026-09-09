"""Cek apakah pola leksikal bisa dideteksi dengan regex."""
import json
import re

POLA = re.compile(
    r"pasal\s+\d+|lampiran\s+huruf\s+[a-z]|"
    r"pmk\s+\d+|pp\s+\d+|nomor\s+\d+|angka\s+\d+",
    re.IGNORECASE,
)

soal = [json.loads(l) for l in open("eval/golden_set.jsonl", encoding="utf-8")]

print(f"{'id':6s} {'kategori':14s} {'terdeteksi':11s} pertanyaan")
print("-" * 80)
for q in soal:
    ada = bool(POLA.search(q["question"]))
    tanda = "LEKSIKAL" if ada else ""
    print(f"{q['id']:6s} {q['category']:14s} {tanda:11s} {q['question'][:44]}")

print()
lex = [q for q in soal if q["category"] == "lexical"]
kena = [q for q in lex if POLA.search(q["question"])]
salah = [q for q in soal if q["category"] != "lexical" and POLA.search(q["question"])]
print(f"soal lexical terdeteksi : {len(kena)}/{len(lex)}")
print(f"salah deteksi (non-lex) : {len(salah)}")
for q in salah:
    print(f"  {q['id']} ({q['category']}): {q['question'][:50]}")