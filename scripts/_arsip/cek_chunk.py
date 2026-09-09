import json
import collections

c = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]

print("=== 5 chunk terpendek ===")
for k in sorted(c, key=lambda x: len(x["text"]))[:5]:
    print(f"{len(k['text']):4d} | {k['pasal'][:22]:22s} | {k['text'][:70]}")

print()
print("=== chunk tanpa pasal ===")
tanpa = [x for x in c if not x["pasal"]]
print("jumlah:", len(tanpa))
for x in tanpa[:8]:
    print(f"  {x['bab'][:42]:42s} | {x['text'][:45]}")

print()
print("=== sebaran ===")
print(collections.Counter(x["doc"] for x in c))

print()
print("=== contoh chunk tabel TER ===")
for x in c:
    if "Lampiran huruf A" in (x["bab"] + x["pasal"]):
        print("bab  :", x["bab"][:50])
        print("pasal:", x["pasal"][:50])
        print("isi  :", x["text"][:250].replace("\n", " "))
        break