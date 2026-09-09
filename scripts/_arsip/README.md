# Arsip

Skrip diagnosis dan pendekatan yang ditinggalkan. Disimpan karena
menjadi bukti untuk klaim di README utama.

## Pendekatan yang digantikan library

| File | Digantikan oleh |
|---|---|
| `parse_basic.py` | `pymupdf4llm.to_markdown()` — 1 baris menggantikan ~30 |
| `normalize.py` | Tidak perlu setelah PDF ditulis ulang. Sempat punya 2 bug: regex `\bo/o` tidak pernah cocok, dan deteksi tabel hanya menangkap halaman pertama |
| `retriever.py` | Dipindahkan ke `shared/retrieval.py` |

## Eksperimen yang menghasilkan temuan

| File | Temuan |
|---|---|
| `uji_bobot.py` | Query leksikal butuh sparse 0,7+; query semantik butuh dense 0,7+. Berlawanan, tidak ada satu bobot yang memuaskan keduanya |
| `uji_topn.py`, `uji_jumlah_chunk.py` | Menambah konteks menurunkan kualitas jawaban pada model 4B |
| `uji_stabil.py` | Memastikan hasil reproducible, bukan variasi acak |
| `uji_halusinasi.py` | `top_n=4` menghasilkan halusinasi pada 2 jalan berturut-turut |
| `uji_mode.py` | Dense dan hybrid identik (0,640) pada 3 jalan masing-masing |
| `uji_bm25_token.py` | Membuktikan kata umum ("PP", "58", "Tahun") menenggelamkan sinyal pembeda BM25 |

## Skrip diagnosis sekali pakai

`cek_chunk.py`, `cek_goldenset.py`, `cek_match.py`, `embed_test.py`,
`lihat_konteks.py`, `lihat_chunk_pasal3.py`, `uji_search.py`,
`uji_shared.py`, `uji_langfuse.py`, `uji_router.py`, `uji_routed.py`,
`uji_rerank.py`, `uji_hybrid.py`

Sebagian memakai jalur lama `~/ai-projects/`, jadi tidak jalan langsung
tanpa penyesuaian.