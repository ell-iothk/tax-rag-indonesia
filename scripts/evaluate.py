"""Ukur kualitas retrieval terhadap golden set. Metrik dihitung ranx."""

import json
import re
import sys
from pathlib import Path

from ranx import Qrels, Run, evaluate

from taxrag.retrieval import MODE, get_retriever

GOLDEN = Path("eval/golden_set.jsonl")
HASIL = Path("eval/hasil.json")

mode = sys.argv[1] if len(sys.argv) > 1 else MODE
NAMA_RUN = mode
K = 10
METRIK = ["recall@5", "recall@10", "ndcg@10", "mrr", "precision@5"]


def kunci_dari(src: str) -> str | None:
    """source_section -> kunci yang cocok dengan metadata chunk."""
    m = re.search(r"Pasal \d+", src)
    if m:
        return m.group(0)
    m = re.search(r"Lampiran huruf [A-Z]", src)
    return m.group(0) if m else None


def id_dokumen(meta: dict, kunci: str) -> str:
    """ID stabil untuk satu unit dokumen."""
    return f"{meta.get('doc')}::{kunci}"


def cocok(meta: dict, kunci: str) -> bool:
    if kunci.startswith("Pasal"):
        return meta.get("pasal", "") == kunci  # persis, bukan 'in'
    return kunci in meta.get("bab", "") or kunci in meta.get("pasal", "")


def bangun(store, soal):
    """Hasilkan qrels (jawaban benar) dan run (hasil sistem)."""
    qrels, run = {}, {}
    for q in soal:
        kunci = kunci_dari(q["source_section"])
        qid = q["id"]

        doc_id = q["source_doc"].replace(".pdf", "")
        qrels[qid] = {f"{doc_id}::{kunci}": 1}

        hasil = store.invoke(q["question"])
        skor_per_unit = {}
        for i, doc in enumerate(hasil, 1):
            m = doc.metadata
            unit = (
                id_dokumen(m, kunci)
                if cocok(m, kunci)
                else f"{m.get('doc')}::{m.get('pasal') or m.get('bab') or '?'}"
            )
            # skor = 1/posisi, karena retriever tidak kembalikan skor mentah.
            # satu unit bisa punya beberapa chunk, ambil yang posisinya tertinggi
            skor_per_unit[unit] = max(skor_per_unit.get(unit, 0.0), 1.0 / i)
        run[qid] = skor_per_unit
    return qrels, run


if __name__ == "__main__":
    store = get_retriever("tax_docs", "data/processed/chunks.jsonl", mode=mode, top_n=K)

    soal = [json.loads(baris) for baris in open(GOLDEN, encoding="utf-8")]
    soal = [q for q in soal if q["category"] != "unanswerable"]

    qrels_d, run_d = bangun(store, soal)
    hasil = evaluate(Qrels(qrels_d), Run(run_d, name=NAMA_RUN), METRIK)

    print("=" * 58)
    print(f"HASIL: {NAMA_RUN}  (n={len(soal)} soal)")
    print("=" * 58)
    for m in METRIK:
        print(f"  {m:14s} = {hasil[m]:.4f}")

    # per kategori
    print("\nper kategori:")
    for kat in sorted({q["category"] for q in soal}):
        sub = [q["id"] for q in soal if q["category"] == kat]
        h = evaluate(
            Qrels({k: qrels_d[k] for k in sub}),
            Run({k: run_d[k] for k in sub}),
            ["recall@5", "ndcg@10"],
        )
        print(f"  {kat:12s} n={len(sub):2d} | R@5 {h['recall@5']:.3f} | NDCG@10 {h['ndcg@10']:.3f}")

    # yang gagal total
    print("\nGAGAL (jawaban benar tidak masuk top-10):")
    gagal = [q for q in soal if list(qrels_d[q["id"]])[0] not in run_d[q["id"]]]
    for q in gagal:
        print(f"  {q['id']} ({q['category']}) cari: {list(qrels_d[q['id']])[0]}")
    if not gagal:
        print("  (tidak ada)")

    # simpan
    riwayat = json.loads(HASIL.read_text()) if HASIL.exists() else {}
    riwayat[NAMA_RUN] = {m: hasil[m] for m in METRIK}
    HASIL.write_text(json.dumps(riwayat, indent=2), encoding="utf-8")
    print(f"\ndisimpan -> {HASIL}")
