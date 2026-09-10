# Strategi retrieval untuk RAG.
import json
import re
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from taxrag.models import get_embeddings, get_reranker

QDRANT_URL = "http://localhost:6333"

# ---------------------------------------------------------------- konfigurasi
MODE = "dense"  # dense | sparse | hybrid | dense_rerank | hybrid_rerank
TOP_N = 5  # berapa chunk dikirim ke LLM
K_KANDIDAT = 20  # over-fetch sebelum rerank
BOBOT_HYBRID = (0.7, 0.3)  # (dense, sparse)


def muat_docs(path: str | Path) -> list[Document]:
    """Baca chunk dari jsonl. Dibutuhkan BM25 yang bekerja di memori."""
    docs = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        teks = c.pop("text")
        docs.append(Document(page_content=teks, metadata=c))
    return docs


# ==================================================== TAHAP 1: KANDIDAT


def dense_retriever(collection: str, k: int = 10):
    """Bi-encoder (bge-m3) + HNSW. Menangkap MAKNA, bukan kata.

    HNSW = algoritma pencarian, BUKAN model. Dia tidak membandingkan query
    dengan semua vector satu per satu, tapi membangun peta jalan pintas
    berlapis. Hasilnya PERKIRAAN, bukan pasti tepat — trade-off recall
    vs latency.
    """
    store = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        url=QDRANT_URL,
        collection_name=collection,
    )
    return store.as_retriever(search_kwargs={"k": k})


def sparse_retriever(chunks_path: str | Path, k: int = 10):
    """BM25 — pencocokan KATA harfiah, bukan makna.

    Membobot kata berdasarkan frekuensi di dokumen dibagi kelangkaan di
    korpus (IDF). Kata langka seperti "21" atau "selebgram" dapat bobot
    tinggi; kata umum seperti "yang" hampir nol.

    Bekerja di memori, tidak butuh database. Untuk korpus besar,
    ganti ke sparse vector di Qdrant.
    """
    r = BM25Retriever.from_documents(muat_docs(chunks_path))
    r.k = k
    return r


def hybrid_retriever(
    collection: str,
    chunks_path: str | Path,
    k: int = 10,
    bobot=BOBOT_HYBRID,
    k_kandidat: int | None = None,
):
    """Gabung dense + sparse dengan RRF (Reciprocal Rank Fusion).

    RRF: skor(dok) = sum(1 / (60 + posisi_di_daftar_i))

    Memakai POSISI, bukan skor mentah. Alasannya: skor cosine (0-1) dan
    skor BM25 (0-tak terbatas) tidak sebanding, menjumlahkannya tidak
    bermakna. RRF membuat kedua daftar setara tanpa normalisasi.
    Ini pertanyaan interview yang sering muncul.

    KELEMAHAN TERUKUR: RRF menghukum dokumen yang cuma kuat di SATU sisi.
    Padahal justru itu yang dibutuhkan query leksikal (q028). Menaikkan
    k_kandidat malah memperburuk, karena dokumen mediocre yang muncul di
    kedua daftar mengalahkan dokumen bagus yang kuat di satu daftar.
    """
    kk = k_kandidat or k
    return EnsembleRetriever(
        retrievers=[dense_retriever(collection, kk), sparse_retriever(chunks_path, kk)],
        weights=list(bobot),
    )


# ==================================================== TAHAP 2: RERANK


def rerank(base_retriever, top_n: int = TOP_N):
    """Cross-encoder mengurutkan ulang kandidat.

    BI-ENCODER (bge-m3): embed query dan dokumen TERPISAH, bandingkan
    vector. Dokumen bisa di-embed sekali di awal -> cepat.

    CROSS-ENCODER (bge-reranker): baca query DAN dokumen BERSAMAAN,
    keluarkan satu skor. Bisa melihat interaksi antar-kata -> akurat.
    Tapi tidak bisa dihitung di awal, harus per pasangan -> lambat.

    Pola dua tahap: bi-encoder saring banyak (murah), cross-encoder
    pilih sedikit (mahal). "Over-fetch lalu saring."

    v2-m3, BUKAN base. bge-reranker-base hanya Inggris/Mandarin dan akan
    gagal DIAM-DIAM pada korpus Indonesia.
    """
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=get_reranker(), top_n=top_n),
        base_retriever=base_retriever,
    )


# ==================================================== PILIHAN AKTIF


class _PotongTopN:
    """Bungkus retriever supaya hasilnya tepat top_n.

    EnsembleRetriever mengembalikan gabungan unik dua daftar, jumlahnya
    bisa lebih dari k. Kontrak dengan tahap generasi harus pasti supaya
    context window terkendali.
    """

    def __init__(self, retriever, top_n: int):
        self.retriever = retriever
        self.top_n = top_n

    def invoke(self, query: str, **kwargs):
        return self.retriever.invoke(query, **kwargs)[: self.top_n]


def get_retriever(collection: str, chunks_path: str | Path, mode: str = MODE, top_n: int = TOP_N):
    """Kembalikan retriever sesuai mode. Ganti MODE di atas untuk beralih."""

    if mode == "hybrid":
        # >>> TERPILIH <<< R@5 0.800, p95 135ms
        r = hybrid_retriever(collection, chunks_path, top_n, BOBOT_HYBRID, top_n)
        return _PotongTopN(r, top_n)

    if mode == "dense":
        # cadangan. MRR lebih baik (0.630 vs 0.572), R@5 lebih rendah
        return dense_retriever(collection, top_n)

    if mode == "sparse":
        return sparse_retriever(chunks_path, top_n)

    if mode == "routed":
        return routed_retriever(collection, chunks_path, top_n)

    if mode == "dense_rerank":
        # HIDUPKAN KALAU ADA GPU TERPISAH. R@5 0.840, p95 40 detik di CPU
        return rerank(dense_retriever(collection, K_KANDIDAT), top_n)

    if mode == "hybrid_rerank":
        return rerank(
            hybrid_retriever(collection, chunks_path, K_KANDIDAT, BOBOT_HYBRID, K_KANDIDAT), top_n
        )

    raise ValueError(f"mode tidak dikenal: {mode}")


# ==================================================== QUERY ROUTING


POLA_LEKSIKAL = re.compile(
    r"pasal\s+\d+|lampiran\s+huruf\s+[a-z]|"
    r"pmk\s+\d+|pp\s+\d+|nomor\s+\d+|angka\s+\d+",
    re.IGNORECASE,
)


def klasifikasi_query(pertanyaan: str) -> str:
    """Tentukan jenis pertanyaan: leksikal atau semantik.

    LEKSIKAL  = menyebut nomor pasal/lampiran/peraturan secara harfiah.
                Butuh BM25, karena embedding tidak bisa membedakan
                "Pasal 21" dari "Pasal 15" — bagi model semua nomor
                pasal punya makna serupa.
    SEMANTIK  = deskripsi situasi atau konsep. Butuh dense.

    BUKTI PERLUNYA (uji bobot Step 6):
      q028 "Apa isi Pasal 21 ayat (1)?"
          dense 1.0/sparse 0.0 -> TIDAK KETEMU
          dense 0.0/sparse 1.0 -> posisi 5
      q016 "gaji 10 juta kena TER berapa?"
          dense 1.0/sparse 0.0 -> posisi 1
          dense 0.0/sparse 1.0 -> TIDAK KETEMU
    Dua-duanya menginginkan bobot BERLAWANAN. Tidak ada satu
    konfigurasi yang memuaskan keduanya.

    Deteksi pakai regex, bukan LLM. Alasannya: pola penyebutan di
    dokumen hukum sangat khas, dan regex itu deterministik + gratis.
    Uji di src/uji_router.py: 3/3 soal lexical tertangkap, 1 salah
    tangkap (q015, yang memang menyebut "PMK 168/2023" harfiah).
    """
    return "leksikal" if POLA_LEKSIKAL.search(pertanyaan) else "semantik"


class RoutedRetriever:
    """Adaptive RAG: pilih retriever berdasarkan jenis pertanyaan."""

    def __init__(self, retriever_leksikal, retriever_semantik):
        self.leksikal = retriever_leksikal
        self.semantik = retriever_semantik
        self.riwayat = []  # untuk analisis: query mana ke mana

    def invoke(self, query: str, **kwargs):
        jenis = klasifikasi_query(query)
        self.riwayat.append((query, jenis))
        r = self.leksikal if jenis == "leksikal" else self.semantik
        return r.invoke(query, **kwargs)


def routed_retriever(
    collection: str,
    chunks_path,
    top_n: int = TOP_N,
    bobot_leksikal=(0.2, 0.8),
    bobot_semantik=(0.8, 0.2),
):
    """Query leksikal -> sparse dominan. Query semantik -> dense dominan."""
    lex = _PotongTopN(
        hybrid_retriever(collection, chunks_path, top_n, bobot_leksikal, top_n), top_n
    )
    sem = _PotongTopN(
        hybrid_retriever(collection, chunks_path, top_n, bobot_semantik, top_n), top_n
    )
    return RoutedRetriever(lex, sem)


# ==================================================== BELUM DIUJI
#
# Semua di bawah BELUM PERNAH DIJALANKAN. Kerangka saja.
# Aktifkan satu per satu, ukur dengan golden set, catat hasilnya di atas.
#
# ------------------------------------------------------------------
# def mmr_retriever(collection, k=10, fetch_k=30, lambda_mult=0.5):
#     """Maximal Marginal Relevance: seimbangkan RELEVANSI dan KERAGAMAN.
#
#     Masalah yang diselesaikan: 44 baris tarif TER Kategori A bentuk
#     kalimatnya nyaris identik, cuma angkanya beda. Mereka saling
#     berebut posisi dan mendominasi top-5, padahal cuma satu yang benar.
#
#     lambda_mult=1.0 -> relevansi saja (sama dengan similarity biasa)
#     lambda_mult=0.0 -> keragaman maksimal
#
#     store.as_retriever(search_type="mmr", search_kwargs={
#         "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
#     """
#
# ------------------------------------------------------------------
# def metadata_filter_retriever(collection, k=10, **filter):
#     """Saring berdasarkan payload Qdrant SEBELUM pencarian vector.
#
#     Berguna untuk memisahkan batang_tubuh vs penjelasan vs lampiran.
#     Contoh: pertanyaan tentang aturan harus cari di batang_tubuh, bukan
#     di bagian "Cukup jelas" milik penjelasan.
#
#     PENTING: pre-filter (saring dulu, cari kemudian) berbeda dengan
#     post-filter (cari dulu, saring kemudian). Post-filter merusak top-k
#     karena bisa menyisakan kurang dari k hasil. Qdrant mendukung
#     pre-filter, jadi pakai itu.
#
#     from qdrant_client.http import models
#     f = models.Filter(must=[models.FieldCondition(
#         key="metadata.doc", match=models.MatchValue(value="pmk168-2023"))])
#     store.as_retriever(search_kwargs={"k": k, "filter": f})
#     """
#
# ------------------------------------------------------------------
# def query_router(pertanyaan: str, llm) -> str:
#     """ADAPTIVE RAG: pilih strategi berdasarkan jenis pertanyaan.
#
#     BUKTI EMPIRIS PERLUNYA (dari uji bobot Project 1):
#       q028 "Apa isi Pasal 21 ayat (1)?" -> butuh sparse 0.7+
#       q016 "gaji 10 juta kena TER berapa?" -> butuh dense 0.7+
#     Tidak ada satu bobot yang memuaskan keduanya. Routing menyelesaikan
#     ini dengan mengarahkan tiap jenis ke pipeline yang sesuai.
#
#     Best practice 2026: classifier merutekan query berdasarkan
#     kompleksitas — sederhana ke pipeline cepat, kompleks ke agentic.
#
#     prompt = "Klasifikasikan pertanyaan ini:
#               LEKSIKAL - menyebut nomor pasal/kode/istilah persis
#               SEMANTIK - deskripsi situasi atau konsep
#               HITUNG   - butuh perhitungan angka
#               Jawab satu kata."
#     """
#
# ------------------------------------------------------------------
# def hyde_retriever(pertanyaan: str, llm, store):
#     """HyDE - Hypothetical Document Embeddings.
#
#     LLM mengarang jawaban HIPOTETIS dulu, lalu jawaban itu yang di-embed
#     dan dicari, bukan pertanyaannya.
#
#     Alasannya: pertanyaan dan dokumen punya bentuk berbeda. "Berapa
#     biaya jabatan?" tidak mirip dengan "Besarnya biaya jabatan
#     ditetapkan sebesar 5%..." secara embedding. Tapi jawaban karangan
#     LLM bentuknya mirip dokumen.
#
#     Biaya: satu panggilan LLM per query. Menambah latensi signifikan.
#
#     jawaban_palsu = llm.invoke(f"Tulis jawaban singkat untuk: {pertanyaan}")
#     return store.similarity_search(jawaban_palsu.content, k=10)
#     """
#
# ------------------------------------------------------------------
# def multi_query_retriever(store, llm, k=10):
#     """LLM membuat 3 parafrase pertanyaan, cari semuanya, gabung hasilnya.
#
#     Mengatasi masalah bahwa satu cara bertanya mungkin tidak cocok
#     dengan bahasa dokumen. Tiga versi menaikkan peluang salah satunya
#     cocok.
#
#     from langchain.retrievers.multi_query import MultiQueryRetriever
#     return MultiQueryRetriever.from_llm(
#         retriever=store.as_retriever(search_kwargs={"k": k}), llm=llm)
#     """
#
# ------------------------------------------------------------------
# def sparse_vector_qdrant(collection, k=10):
#     """BM25/SPLADE sebagai sparse vector DI DALAM Qdrant, bukan di memori.
#
#     BM25Retriever memuat seluruh korpus ke RAM. Untuk 91 chunk tidak
#     masalah; untuk 1 juta chunk tidak mungkin. Qdrant mendukung sparse
#     vector natif, sehingga hybrid bisa dilakukan dalam satu query
#     ke database.
#
#     bge-m3 menghasilkan dense DAN sparse vector sekaligus — keunggulan
#     yang belum kita pakai.
#
#     from langchain_qdrant import FastEmbedSparse, RetrievalMode
#     QdrantVectorStore.from_existing_collection(
#         embedding=get_embeddings(),
#         sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
#         retrieval_mode=RetrievalMode.HYBRID, ...)
#     """
