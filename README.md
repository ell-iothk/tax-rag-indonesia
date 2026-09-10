# Tax RAG Indonesia

Sistem tanya-jawab dokumen perpajakan Indonesia yang berjalan **sepenuhnya offline** di laptop dengan GPU 6 GB. Tanpa API cloud, tanpa data keluar dari mesin.

> **English summary** — A fully local RAG system for Indonesian tax regulations, running on a single 6 GB consumer GPU. Built with an evaluation-first approach: every retrieval technique was measured against a hand-written golden set before being kept or discarded. Four techniques were tested and rejected based on data, including cross-encoder reranking (211× latency for 2% NDCG gain) and query routing (no measurable improvement).

---

## Hasil

| Metrik | Nilai |
|---|---|
| Recall@5 | 0,800 |
| Recall@10 | 0,920 |
| NDCG@10 | 0,7068 |
| MRR | 0,6390 |
| Akurasi jawaban | 0,640 |
| Refusal rate | 1,000 |
| **Halusinasi** | **0,000** |
| Menolak padahal bisa | 0,120 |
| Latensi retrieval (p95) | 190 ms |
| Latensi total per pertanyaan | 2,7 detik |

Diukur pada golden set 30 pertanyaan yang ditulis manual. Metrik retrieval dihitung dengan [ranx](https://github.com/AmenRa/ranx), termasuk uji signifikansi statistik.

**Halusinasi 0,000** artinya: dari 5 pertanyaan yang jawabannya sengaja tidak ada di dokumen, sistem menolak menjawab kelimanya. Di domain regulasi, ini lebih penting daripada akurasi.

---

## Yang membedakan proyek ini

Empat teknik diuji lalu **ditolak berdasarkan data**, bukan dipasang karena populer:

| Teknik | Hasil pengukuran | Keputusan |
|---|---|---|
| Cross-encoder reranking | NDCG +2%, latensi **211×** (190 ms → 40.167 ms) | Ditolak |
| Query routing | NDCG turun 0,7068 → 0,6447 | Ditolak |
| Hybrid search (BM25 + RRF) | Setara dense, NDCG lebih rendah | Ditolak |
| Ragas faithfulness | Timeout 100% (25/25 soal) di GPU 6 GB | Ditolak |

Kodenya tetap ada dan dinonaktifkan, lengkap dengan catatan kapan keputusannya harus dibalik — misalnya reranker layak dipakai kalau ada GPU terpisah.

---

## Arsitektur

```
INGESTION (sekali)
  PDF → PyMuPDF4LLM → Markdown → structure-aware chunking
      → bge-m3 (CPU) → Qdrant

QUERY (tiap pertanyaan)
  pertanyaan → bge-m3 (CPU)     ~120 ms
             → HNSW di Qdrant    ~15 ms   → 5 potongan
             → prompt + grounding
             → Qwen3 4B (GPU)  ~2.400 ms  → jawaban + sitasi
```

| Lapisan | Pilihan | Perangkat |
|---|---|---|
| Parsing | PyMuPDF4LLM | CPU |
| Chunking | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | CPU |
| Embedding | BAAI/bge-m3, 1024 dimensi | CPU |
| Vector DB | Qdrant (HNSW) | Docker |
| Generator | Qwen3 4B Instruct Q4_K_M via Ollama | **GPU** |
| Orkestrasi | LangChain LCEL | — |
| Evaluasi | ranx + metrik deterministik | — |
| Observability | Langfuse (self-host) | Docker |

**Pembagian perangkat disengaja.** VRAM 6 GB hanya cukup untuk satu model. Embedding dijalankan di CPU karena sifatnya batch dan tidak butuh real-time; GPU disediakan penuh untuk model bahasa.

---

## Korpus

- **PMK 168/2023** — Petunjuk Pelaksanaan Pemotongan PPh Pasal 21
- **PP 58/2023** — Tarif Pemotongan PPh Pasal 21 (tabel TER)

88 chunk, rata-rata 325 token, maksimum 538.

PDF asli dari JDIH Kemenkeu ditulis ulang menjadi versi bersih karena PP 58 adalah hasil scan dengan OCR rusak — 102 kemunculan `o/o` yang seharusnya `%`, dan angka `0` terbaca sebagai huruf `O`. PDF asli disimpan di `data/raw_asli/` sebagai arsip.

**Temuan yang memungkinkan pemulihan:** dokumen hukum Indonesia menulis angka dua kali — dalam angka dan dalam huruf di dalam kurung. Versi hurufnya utuh sempurna, karena OCR lebih andal membaca huruf daripada digit. Konversi dari versi huruf bersifat deterministik, tanpa tebakan.

Tabel TER (127 baris tarif) diverifikasi terhadap **8 contoh perhitungan resmi** yang ada di PMK 168. Delapan dari delapan cocok.

---

## Menjalankan

### Prasyarat

- WSL2 Ubuntu atau Linux
- GPU NVIDIA dengan VRAM ≥ 5 GB
- Docker
- [Ollama](https://ollama.com)
- Python 3.10+

### Setup

```bash
git clone https://github.com/ell-iothk/tax-rag-indonesia.git
cd tax-rag-indonesia

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
> `-e` berarti *editable install* — proyek dipasang sebagai tautan, bukan
> salinan, sehingga perubahan kode langsung berlaku tanpa install ulang.
> Ini yang membuat `from taxrag.retrieval import ...` bekerja dari folder
> mana pun, tanpa perlu mengutak-atik `sys.path`.
>
> Model embedding (bge-m3, 2,2 GB) dan reranker (bge-reranker-v2-m3, 2,2 GB)
> diunduh otomatis saat pertama dijalankan ke `.hf-cache/`. Sediakan ~5 GB
> ruang disk. Kalau sudah punya cache HuggingFace di tempat lain:
> `export HF_HOME=/jalur/ke/cache`

Siapkan model bahasa:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create qwen3-id -f Modelfile.qwen3-id
ollama ps        # pastikan "100% GPU" dengan CONTEXT 8192
```
Unduh model embedding dan reranker:

```bash
python scripts/download_models.py
```

Sekitar 4,4 GB ke folder `models/`, sekali saja. Model disimpan sebagai
file biasa, bukan cache HuggingFace — struktur cache mudah rusak kalau ada
file terhapus manual dan sulit didiagnosis.
Nyalakan database:

```bash
cp .env.example .env      # isi secret, lihat komentar di dalamnya
docker compose up -d
```

Bangun index:

```bash
python scripts/ingest.py          # PDF → 88 chunk
python scripts/index_qdrant.py    # chunk → Qdrant (2–5 menit di CPU)
```

### Pakai

```bash
./rag.sh
```

```
tanya> Biaya jabatan maksimal berapa setahun?

Biaya jabatan maksimal sebesar Rp6.000.000 setahun
(potongan [1], PMK 168/2023 Pasal 10 ayat (2)).

  sumber (2.4s): pmk168-2023 Pasal 10 | pmk168-2023 Pasal 11
  /sumber untuk lihat isi lengkapnya
```

Perintah CLI: `/sumber` menampilkan potongan yang dipakai, `/keluar` untuk selesai.

### Evaluasi ulang

```bash
python scripts/evaluate.py              # metrik retrieval
python scripts/evaluate.py hybrid       # bandingkan konfigurasi lain
python scripts/evaluate_jawaban.py      # metrik kualitas jawaban
python scripts/ukur_latensi.py          # p50 dan p95
```

---

## Metodologi evaluasi

### Golden set ditulis manual

30 pertanyaan, dengan komposisi yang dirancang untuk menguji komponen berbeda:

| Kategori | n | Menguji |
|---|---|---|
| `single_hop` | 15 | Retrieval dasar |
| `multi_hop` | 7 | Penggabungan beberapa sumber — di sini chunking ketahuan |
| `unanswerable` | 5 | Keberanian menolak — pengukur halusinasi |
| `lexical` | 3 | Kecocokan kata harfiah (nomor pasal) |

Pertanyaan **tidak digenerate LLM**. Kalau pertanyaan dibuat dari potongan yang sudah ada di index, evaluasinya menjadi sirkular dan skornya tinggi secara palsu.

Field `source_section` memungkinkan **pengukuran retrieval terpisah dari generation**. Kalau jawaban salah, bisa ditentukan apakah penyebabnya pengambilan yang keliru atau penyimpulan yang keliru.

### Kenapa tidak memakai Ragas untuk retrieval

Ragas memerlukan LLM sebagai penilai, sedangkan `source_section` adalah ground truth pasti. Penilaian deterministik lebih cepat, konsisten antar-jalan, dan bebas dari bias LLM-as-judge (*position bias*, *verbosity bias*, *self-preference*).

Ragas tetap dicoba untuk mengukur *faithfulness*, dan gagal karena timeout — lihat bagian batasan.

---

## Beberapa temuan

### Menambah konteks bisa menurunkan kualitas jawaban

Diuji tiga kali per konfigurasi, semuanya reproducible:

| `top_n` | q010 "siapa tenaga ahli?" | Halusinasi |
|---|---|---|
| 3 | benar 3/3 | 0/5 |
| 4 | benar 3/3 | **1/5** |
| 5 | **salah 3/3** | 0/5 |

Potongan ke-5 tidak relevan dan membelokkan model dari jawaban yang ada di potongan ke-1. Dipilih `top_n=5` karena halusinasi lebih berbahaya daripada satu soal akurasi di domain regulasi.

### Metrik retrieval hanya proksi

Dense unggul di NDCG (0,7068 versus 0,6595) dan MRR, tetapi kualitas jawabannya identik dengan hybrid — 0,640 pada tiga jalan masing-masing. Keputusan retrieval harus diverifikasi terhadap keluaran akhir, bukan berhenti di metrik perantara.

### Skor embedding tidak boleh dipakai sebagai ambang mutlak

Jawaban benar untuk soal q002 mendapat skor cosine 0,4042; query yang sama sekali tidak relevan mendapat 0,3578. Selisihnya 0,05. Model multilingual memberi skor dasar tinggi untuk teks berbahasa Indonesia apa pun.

### Prompt engineering punya efek samping

Instruksi "baca semua potongan" berhasil menghilangkan halusinasi dari 0,200 menjadi 0,000, tetapi menyebabkan regresi pada satu soal lain. Setiap perubahan prompt diukur terhadap seluruh golden set, bukan satu-dua contoh.

---

## Batasan

| Batasan | Penjelasan |
|---|---|
| **Golden set 25 soal terjawab** | Terlalu kecil untuk uji signifikansi yang kuat. Selisih 2 soal adalah noise |
| **Reranker tidak dipakai** | Batasan VRAM, bukan batasan teknik. Dengan GPU terpisah, Recall@5 naik ke 0,840 |
| **Ragas faithfulness gagal** | Butuh 3–5 panggilan LLM berurutan per soal; timeout 180 detik terlampaui di 100% soal |
| **Kategori lexical 0/3** | Query yang menyebut nomor pasal harfiah masih gagal. Kata umum seperti "PP", "58", "Tahun" muncul di hampir semua potongan sehingga BM25 kehilangan sinyal pembeda |
| **Model 4B** | Salah memilih item dari daftar bertingkat (q010). Terverifikasi bukan masalah retrieval — jawabannya ada di potongan pertama |
| **PDF ditulis ulang** | Bukan salinan resmi. Untuk keperluan nyata, rujuk PDF asli di `data/raw_asli/` |

---

## Struktur

```
.
.
├── pyproject.toml        metadata + dependensi (PEP 621)
├── src/
│   └── taxrag/           package inti, di-import
│       ├── models.py     pemuat model dengan cache
│       ├── chunking.py   pipeline chunking + teknik lain (dikomentari)
│       ├── retrieval.py  dense/sparse/hybrid/rerank/routed
│       └── rag.py        LCEL chain
├── scripts/              entry point, dijalankan langsung
│   ├── ingest.py         PDF → chunk
│   ├── index_qdrant.py   chunk → Qdrant
│   ├── tanya.py          CLI interaktif
│   ├── evaluate*.py      evaluator
│   └── _arsip/           skrip diagnosis dan pendekatan yang ditinggalkan
├── data/
│   ├── raw/              PDF bersih, input pipeline
│   ├── raw_asli/         PDF asli dari JDIH (tidak masuk git)
│   └── processed/        chunks.jsonl (dihasilkan, tidak masuk git)
├── eval/
│   ├── golden_set.jsonl  30 soal, ditulis manual
│   ├── hasil.json        metrik retrieval per konfigurasi
│   └── hasil_jawaban.json
├── docker-compose.yml
├── Modelfile.qwen3-id
└── rag.sh                startup satu perintah
```

Pemisahan `src/` dan `scripts/` mengikuti **src-layout** yang direkomendasikan
PyPA: `src/taxrag/` berisi kode yang di-*import*, `scripts/` berisi kode yang
di-*jalankan*. Karena akar repo tidak berisi package apa pun, satu-satunya cara
import bekerja adalah benar-benar meng-install — sehingga kesalahan setup
ketahuan di mesin sendiri, bukan di mesin orang lain.


Teknik yang belum diuji ditulis sebagai kerangka berkomentar di `src/taxrag/chunking.py` dan `src/taxrag/retrieval.py`, lengkap dengan penjelasan cara kerja: parent-child chunking, semantic chunking, late chunking, MMR, metadata pre-filter, HyDE, multi-query.

---

## Lisensi

MIT
