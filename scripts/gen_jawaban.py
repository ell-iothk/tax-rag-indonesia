"""Tahap 1: jawab semua soal, simpan ke file. Hakim belum ikut."""
import json
from pathlib import Path

from tqdm import tqdm

from taxrag.rag import buat_chain

GOLDEN = Path("eval/golden_set.jsonl")
OUT = Path("eval/jawaban_untuk_ragas.json")

if __name__ == "__main__":
    chain = buat_chain()
    soal = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]
    soal = [q for q in soal if q["category"] != "unanswerable"]

    data = []
    for q in tqdm(soal, desc="menjawab"):
        out = chain.invoke({"pertanyaan": q["question"]})
        data.append({
            "id": q["id"],
            "user_input": q["question"],
            "response": out["jawaban"].strip(),
            "retrieved_contexts": [d.page_content for d in out["docs"]],
            "reference": q["answer"],
        })

    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{len(data)} jawaban -> {OUT}")