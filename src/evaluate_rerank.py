"""Bandingkan dense, hybrid, dan varian rerank pada golden set penuh."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

from ranx import Qrels, Run, compare

from evaluate_hybrid import kunci_dari, bangun_run
from shared.retrieval import (dense_retriever, hybrid_retriever,
                       dense_rerank_retriever, rerank)

GOLDEN = Path("eval/golden_set.jsonl")
K = 10
METRIK = ["recall@5", "recall@10", "ndcg@10", "mrr"]

if __name__ == "__main__":
    soal = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]
    soal = [q for q in soal if q["category"] != "unanswerable"]

    qrels = {
        q["id"]: {f"{q['source_doc'].replace('.pdf','')}::{kunci_dari(q['source_section'])}": 1}
        for q in soal
    }

    konfig = {
        "dense":              dense_retriever(K),
        "hybrid_70_30":       hybrid_retriever(K, (0.7, 0.3), K),
        "rerank_k20":         dense_rerank_retriever(top_n=K, k_kandidat=20),
        "rerank_k30":         dense_rerank_retriever(top_n=K, k_kandidat=30),
        "hybrid_rerank":      rerank(hybrid_retriever(20, (0.7, 0.3), 20), top_n=K),
    }

    runs = []
    for i, (nama, r) in enumerate(konfig.items(), 1):
        print(f"[{i}/{len(konfig)}] {nama}")
        runs.append(Run(bangun_run(r, soal, label=nama), name=nama))

    print()
    print(compare(qrels=Qrels(qrels), runs=runs, metrics=METRIK, max_p=0.05))