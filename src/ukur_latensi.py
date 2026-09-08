"""Ukur latensi tiap konfigurasi retriever."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import statistics
import time
from pathlib import Path

from shared.retrieval import (dense_retriever, hybrid_retriever,
                       dense_rerank_retriever)

soal = [json.loads(l) for l in open("eval/golden_set.jsonl", encoding="utf-8")]
soal = [q["question"] for q in soal if q["category"] != "unanswerable"]

konfig = {
    "dense":        dense_retriever(10),
    "hybrid_70_30": hybrid_retriever(10, (0.7, 0.3), 10),
    "rerank_k20":   dense_rerank_retriever(top_n=5, k_kandidat=20),
}

print(f"{'konfigurasi':16s} {'rata':>8s} {'p50':>8s} {'p95':>8s}")
print("-" * 44)

for nama, r in konfig.items():
    r.invoke(soal[0])          # pemanasan, jangan dihitung
    waktu = []
    for q in soal:
        t0 = time.perf_counter()
        r.invoke(q)
        waktu.append((time.perf_counter() - t0) * 1000)
    waktu.sort()
    p50 = statistics.median(waktu)
    p95 = waktu[int(len(waktu) * 0.95)]
    print(f"{nama:16s} {statistics.mean(waktu):7.0f}ms {p50:7.0f}ms {p95:7.0f}ms")