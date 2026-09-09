"""Uji q010 di beberapa top_n, ulang 3x untuk lihat konsistensi."""
from rag import buat_chain

Q = "Siapa saja yang dianggap tenaga ahli?"

for n in [3, 4, 5]:
    chain = buat_chain(top_n=n)
    print(f"--- top_n={n} ---")
    for i in range(3):
        out = chain.invoke({"pertanyaan": Q})
        jw = out["jawaban"].strip()
        benar = "pengacara" in jw.lower()
        print(f"  [{i+1}] {'BENAR' if benar else 'SALAH'} | {jw[:75]}")
    print()