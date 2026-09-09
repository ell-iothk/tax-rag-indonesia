"""Cek apakah source_section golden set bisa dicocokkan ke metadata chunk."""
import json
import re
from collections import Counter

# semua nilai 'pasal' yang ada di chunk
chunk_pasal = set()
chunk_bab = set()
for line in open("data/processed/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    if c["pasal"]:
        chunk_pasal.add(c["pasal"])
    if c["bab"]:
        chunk_bab.add(c["bab"])

print("=== nilai 'pasal' di chunk ===")
for p in sorted(chunk_pasal):
    print(" ", p)

print()
print("=== source_section di golden set ===")
cocok, tidak = 0, []
for line in open("eval/golden_set.jsonl", encoding="utf-8"):
    q = json.loads(line)
    if q["category"] == "unanswerable":
        continue
    src = q["source_section"]
    # ambil "Pasal N" dari string seperti "Pasal 10 ayat (2)"
    m = re.search(r"Pasal \d+", src)
    if m:
        kunci = m.group(0)
        ada = kunci in chunk_pasal
    else:
        m2 = re.search(r"Lampiran huruf [A-Z]", src)
        kunci = m2.group(0) if m2 else None
        ada = any(kunci in b for b in chunk_bab) if kunci else False
    if ada:
        cocok += 1
    else:
        tidak.append((q["id"], src, kunci))
    print(f"  {q['id']} | {src[:42]:44s} -> {kunci or '(bukan pasal)':15s} {'OK' if ada else 'X'}")

print()
print(f"cocok: {cocok}, tidak: {len(tidak)}")
print()
print("=== yang tidak cocok ===")
for i, src, k in tidak:
    print(f"  {i}: {src}")



































































