"""Uji model embedding: unduh, cek dimensi, cek pemahaman makna."""
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 1. cek dimensi ---
v = emb.embed_query("Berapa tarif pemotongan PPh 21?")
print("dimensi        :", len(v))
print("5 angka pertama:", [round(x, 4) for x in v[:5]])
print()

# --- 2. cek pemahaman makna ---
kalimat = [
    "Berapa potongan pajak gaji karyawan?",
    "Tarif pemotongan Pajak Penghasilan Pasal 21",
    "Biaya jabatan paling banyak Rp6.000.000 setahun",
    "Resep rendang daging sapi khas Padang",
]

m = np.array(emb.embed_documents(kalimat))

print("kemiripan terhadap:", kalimat[0])
for i in range(1, len(kalimat)):
    skor = float(m[0] @ m[i])
    print(f"  [{i}] {kalimat[i][:45]:47s} = {skor:.4f}")