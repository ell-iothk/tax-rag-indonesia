"""Apakah jumlah chunk mempengaruhi kualitas jawaban?"""
from rag import buat_chain

Q = "Siapa saja yang dianggap tenaga ahli?"

for n in [1, 2, 3, 5]:
    chain = buat_chain(top_n=n)
    out = chain.invoke({"pertanyaan": Q})
    benar = "pengacara" in out["jawaban"].lower()
    print(f"top_n={n} {'BENAR' if benar else 'SALAH'}")
    print(f"  {out['jawaban'].strip()[:100]}")
    print()