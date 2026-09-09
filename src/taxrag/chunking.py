"""
Strategi chunking untuk RAG.

Yang TERUKUR sudah dijalankan dan ada angkanya.
Yang BELUM DIUJI dikomentari — kerangka + penjelasan, aktifkan saat perlu.

HASIL DI PROJECT 1 (korpus perpajakan, 2 PDF, 34 halaman)
----------------------------------------------------------
markdown_header + recursive : 91 chunk, rata 325 token, maks 538
                              Recall@5 0.760, NDCG@10 0.699 (dense)

Parameter yang dipakai dan alasannya:
  CHUNK_SIZE = 1800 karakter (~500 token)
    Benchmark 2026 menunjukkan 512 token adalah default terbaik.
    Di atas ~2500 token ada "context cliff" — kualitas jatuh.
  CHUNK_OVERLAP = 270 (15%)
    Studi Jan 2026 (SPLADE) menemukan overlap tidak selalu berguna.
    Dipertahankan karena kita pakai dense retrieval, bukan sparse.
  MIN_CHAR = 60
    Awalnya 120, tapi itu membuang Pasal 25 ("mulai berlaku pada
    tanggal 1 Januari 2024") dan menggagalkan soal q008.
    60 cukup untuk membuang "Cukup jelas." tapi menyelamatkan pasal pendek.
  headers_to_split_on hanya sampai "###"
    "####" sengaja tidak dipakai: ayat bercetak tebal salah dikenali
    sebagai heading oleh pymupdf4llm.
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 270
MIN_CHAR = 60
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ==================================================== PARSING

def pdf_ke_markdown(pdf_path: str | Path) -> str:
    """PDF -> markdown berstruktur heading.

    TERUKUR: pymupdf4llm menghasilkan 68.749 karakter untuk PMK 168,
    sedangkan pdfplumber 135.002 karakter (56% di antaranya spasi tata
    letak). pymupdf4llm juga menandai heading otomatis, sehingga tidak
    perlu regex untuk mendeteksi "Pasal 10".
    """
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)


# ==================================================== CHUNKING (TERUKUR)

def markdown_header_chunking(md: str, level: int = 3) -> list[Document]:
    """Potong di batas heading. Metadata terisi otomatis dari judul heading.

    Ini structure-aware chunking: mengikuti struktur dokumen, bukan
    memotong buta per N karakter. Menang di dokumen berstruktur kuat
    (hukum, teknis, manual).
    """
    headers = [("#", "judul"), ("##", "bab"), ("###", "pasal")][:level]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,   # judul tetap di teks, membantu retrieval
    )
    return splitter.split_text(md)


def recursive_chunking(docs: list[Document],
                       size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Pecah bagian yang masih terlalu panjang, di batas alami.

    Dipakai sebagai TAHAP 2 setelah markdown_header_chunking.
    Pola dua tahap: struktur dulu, ukuran belakangan.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
    )
    return splitter.split_documents(docs)


def tambah_konteks(doc: Document, doc_id: str) -> Document:
    """Tempel jalur heading di depan teks chunk.

    Ini versi murah dari Contextual Retrieval (Anthropic 2024). Versi
    aslinya memakai LLM untuk menulis ringkasan konteks tiap chunk;
    kita pakai jalur heading yang sudah ada. Untuk dokumen berstruktur
    hasilnya sebanding dengan biaya nol.

    Hasilnya: "[pp58-2023 | Lampiran huruf A] - Kategori A, ..."
    Chunk jadi bisa berdiri sendiri tanpa kehilangan konteks.
    """
    m = doc.metadata
    jalur = " > ".join(x for x in [m.get("bab"), m.get("pasal")] if x)
    isi = re.sub(r"\*+", "", doc.page_content).strip()   # buang markup tebal
    doc.page_content = f"[{doc_id} | {jalur}]\n{isi}" if jalur else isi
    return doc


def bersih_metadata(doc: Document, doc_id: str) -> dict:
    """Rapikan metadata: buang markup, tambahkan id dokumen."""
    b = lambda s: re.sub(r"\*+", "", s or "").strip()
    return {
        "doc": doc_id,
        "judul": b(doc.metadata.get("judul")),
        "bab": b(doc.metadata.get("bab")),
        "pasal": b(doc.metadata.get("pasal")),
    }


def pipeline_standar(pdf_path: str | Path) -> list[Document]:
    """Pipeline yang dipakai Project 1. PDF -> chunk siap embed."""
    pdf_path = Path(pdf_path)
    md = pdf_ke_markdown(pdf_path)
    bagian = markdown_header_chunking(md)
    docs = recursive_chunking(bagian)

    hasil = []
    for d in docs:
        if len(d.page_content.strip()) < MIN_CHAR:
            continue
        if not d.metadata.get("bab") and not d.metadata.get("pasal"):
            continue
        meta = bersih_metadata(d, pdf_path.stem)
        d.metadata = meta
        hasil.append(tambah_konteks(d, pdf_path.stem))
    return hasil


# ==================================================== BELUM DIUJI
#
# Semua di bawah ini BELUM PERNAH DIJALANKAN. Kerangka saja.
# Aktifkan satu per satu, ukur dengan golden set, catat hasilnya di atas.
#
# ------------------------------------------------------------------
# def fixed_size_chunking(teks: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     """Potong per N karakter, abaikan struktur. BASELINE PEMBANDING.
#
#     Gunanya bukan untuk dipakai, tapi untuk membuktikan bahwa
#     structure-aware memang lebih baik. Tanpa baseline ini, klaim
#     "saya pakai structure-aware chunking" tidak punya pembanding.
#     """
#     from langchain_text_splitters import CharacterTextSplitter
#     s = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
#                               separator="")
#     return s.create_documents([teks])
#
# ------------------------------------------------------------------
# def parent_child_chunking(docs, child_size=400, parent_size=2000):
#     """Index chunk KECIL untuk presisi pencarian, kirim chunk BESAR ke LLM.
#
#     Prinsipnya: unit retrieval tidak harus sama dengan unit generasi.
#     Chunk kecil lebih presisi saat dicari (fokus satu ide), tapi terlalu
#     sempit untuk dijawab LLM. Jadi cari yang kecil, kirim induknya.
#
#     RELEVAN UNTUK PROJECT 2 (analyst agent): schema database punya
#     banyak kolom; cari per kolom, kirim seluruh tabel.
#
#     from langchain.retrievers import ParentDocumentRetriever
#     from langchain.storage import InMemoryStore
#     retriever = ParentDocumentRetriever(
#         vectorstore=store,
#         docstore=InMemoryStore(),
#         child_splitter=RecursiveCharacterTextSplitter(chunk_size=child_size),
#         parent_splitter=RecursiveCharacterTextSplitter(chunk_size=parent_size),
#     )
#     """
#
# ------------------------------------------------------------------
# def semantic_chunking(teks, embeddings, threshold_type="percentile"):
#     """Potong di titik LONJAKAN perbedaan makna antar kalimat.
#
#     Cara kerja: embed tiap kalimat, hitung cosine distance antar kalimat
#     berurutan, potong di titik yang jaraknya melebihi ambang persentil.
#
#     PERINGATAN dari benchmark: FloTorch 2026 menemukan semantic chunking
#     menghasilkan fragmen rata-rata 43 token dan hanya mencetak 54%
#     akurasi end-to-end, kalah dari recursive 512 token yang 69%.
#     Mahal (butuh embedding tiap kalimat) dan sering tidak sepadan.
#
#     from langchain_experimental.text_splitter import SemanticChunker
#     return SemanticChunker(embeddings,
#                            breakpoint_threshold_type=threshold_type
#                            ).create_documents([teks])
#     """
#
# ------------------------------------------------------------------
# def late_chunking(teks, model):
#     """Embed SELURUH dokumen dulu di level token, baru potong.
#
#     Menyelesaikan masalah ANAFORA — kata rujukan yang kehilangan acuan
#     saat dipotong. Contoh: chunk 1 "Berlin adalah ibu kota Jerman",
#     chunk 2 "Lebih dari 3,85 juta penduduknya..." — kalau di-embed
#     terpisah, model tidak tahu "-nya" merujuk apa.
#
#     Riset Gunther et al. 2024: naik 10-12% pada dokumen dengan banyak
#     referensi anaforis.
#
#     bge-m3 mendukung ini (context 8192 token). Butuh akses langsung ke
#     token embeddings, jadi tidak bisa lewat HuggingFaceEmbeddings biasa —
#     harus pakai FlagEmbedding atau sentence-transformers langsung.
#     """
#
# ------------------------------------------------------------------
# def contextual_retrieval_llm(chunk, dokumen_penuh, llm):
#     """Versi asli Contextual Retrieval (Anthropic 2024): LLM menulis
#     ringkasan konteks tiap chunk, ditempel sebelum di-embed.
#
#     Lebih kuat dari versi jalur-heading yang kita pakai, tapi biayanya
#     satu panggilan LLM PER CHUNK saat ingestion. Untuk 91 chunk masih
#     wajar; untuk 100.000 chunk mahal.
#
#     prompt = f"Dokumen: {dokumen_penuh[:2000]}
#                Chunk: {chunk}
#                Tulis 1-2 kalimat yang menempatkan chunk ini dalam
#                konteks dokumen. Jawab hanya kalimat konteksnya."
#     konteks = llm.invoke(prompt).content
#     return f"{konteks}\n\n{chunk}"
#     """
#
# ------------------------------------------------------------------
# def ast_chunking(kode: str, bahasa: str = "python"):
#     """Potong kode per fungsi/kelas menggunakan Abstract Syntax Tree.
#
#     UNTUK PROJECT 3 (codebase agent). Memotong kode per N karakter
#     merusak strukturnya; potong per unit sintaksis.
#
#     from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
#     return RecursiveCharacterTextSplitter.from_language(
#         language=Language.PYTHON, chunk_size=1500, chunk_overlap=150
#     ).create_documents([kode])
#     """