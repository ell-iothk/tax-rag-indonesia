"""Unduh model ke folder models/. Jalankan sekali sebelum pemakaian pertama.

Model disimpan sebagai folder biasa, bukan cache HuggingFace. Struktur cache
(blobs + symlink + refs) mudah rusak kalau ada file yang dihapus manual dan
sulit didiagnosis. Folder biasa selalu bisa dilihat isinya dengan ls.

Catatan: bge-m3 hanya menyediakan pytorch_model.bin, tanpa safetensors.
transformers memuatnya sambil menjalankan konversi otomatis di latar
belakang. Itu tidak menghalangi hasil, hanya menambah proses yang berjalan
paralel.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

REPO = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

# bge-m3 hanya menyediakan pytorch_model.bin, tidak ada safetensors.
# Yang dibuang: format kerangka lain, ONNX, dan gambar dokumentasi.
ABAIKAN = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "onnx/*",
    "imgs/*",
    "*.jpg",
    "*.png",
]

if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)

    for nama, repo_id in REPO.items():
        tujuan = MODELS / nama
        print(f"\n=== {repo_id} -> models/{nama} ===")
        snapshot_download(
            repo_id=repo_id,
            local_dir=tujuan,
            ignore_patterns=ABAIKAN,
        )
        ukuran = sum(f.stat().st_size for f in tujuan.rglob("*") if f.is_file())
        print(f"  selesai: {ukuran / 1e9:.2f} GB")

    print("\nlanjut: python scripts/evaluate.py")
