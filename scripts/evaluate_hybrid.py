"""Bandingkan beberapa konfigurasi retriever pada golden set penuh."""

import json
import re
from pathlib import Path

from ranx import Qrels, Run, compare
from retriever import dense_retriever, hybrid_retriever, sparse_retriever
from tqdm import tqdm

GOLDEN = Path("eval/golden_set.jsonl")
HASIL = Path("eval/hasil.json")
K = 10
METRIK = ["recall@5", "recall@10", "ndcg@10", "mrr"]


def kunci_dari(src: str):
    m = re.search(r"Pasal \d+", src)
    if m:
        return m.group(0)
    m = re.search(r"Lampiran huruf [A-Z]", src)
    return m.group(0) if m else None


def cocok(meta: dict, kunci: str) -> bool:
    if kunci.startswith("Pasal"):
        return meta.get("pasal", "") == kunci
    return kunci in meta.get("bab", "") or kunci in meta.get("pasal", "")


def bangun_run(retriever, soal, label: str = "") -> dict:
    """Skor = 1/posisi, karena retriever LangChain tidak selalu kembalikan skor."""
    run = {}
    for q in tqdm(soal, desc=label or "retrieval", leave=False):
        kunci = kunci_dari(q["source_section"])
        hasil = retriever.invoke(q["question"])[:K]
        unit = {}
        for i, d in enumerate(hasil, 1):
            m = d.metadata
            nama = (
                f"{m.get('doc')}::{kunci}"
                if cocok(m, kunci)
                else f"{m.get('doc')}::{m.get('pasal') or m.get('bab') or '?'}"
            )
            unit[nama] = max(unit.get(nama, 0.0), 1.0 / i)
        run[q["id"]] = unit
    return run


if __name__ == "__main__":
    soal = [json.loads(baris) for baris in open(GOLDEN, encoding="utf-8")]
    soal = [q for q in soal if q["category"] != "unanswerable"]

    qrels = {
        q["id"]: {f"{q['source_doc'].replace('.pdf', '')}::{kunci_dari(q['source_section'])}": 1}
        for q in soal
    }

    konfig = {
        "dense": dense_retriever(K),
        "sparse": sparse_retriever(K),
        "hybrid_50_50": hybrid_retriever(K, (0.5, 0.5), K),
        "hybrid_70_30": hybrid_retriever(K, (0.7, 0.3), K),
        "hybrid_30_70": hybrid_retriever(K, (0.3, 0.7), K),
    }

    runs = []
    for nama, r in konfig.items():
        print(f"menjalankan {nama}...")
        runs.append(Run(bangun_run(r, soal), name=nama))

    laporan = compare(
        qrels=Qrels(qrels),
        runs=runs,
        metrics=METRIK,
        max_p=0.05,
    )
    print()
    print(laporan)

    riwayat = json.loads(HASIL.read_text()) if HASIL.exists() else {}
    skor = laporan.results if hasattr(laporan, "results") else laporan.to_dict()
    for run in runs:
        riwayat[run.name] = {m: skor[run.name][m] for m in METRIK}
