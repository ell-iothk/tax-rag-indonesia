"""Tahap 2: nilai jawaban yang sudah tersimpan. Generator tidak dimuat."""
import json
import sys
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithReference, LLMContextRecall)

from shared.models import get_embeddings

IN = Path("eval/jawaban_untuk_ragas.json")
HAKIM = sys.argv[1] if len(sys.argv) > 1 else "qwen3-id"
OUT = Path(f"eval/hasil_ragas_{HAKIM.replace(':', '_')}.json")

if __name__ == "__main__":
    data = json.loads(IN.read_text(encoding="utf-8"))
    print(f"{len(data)} jawaban dimuat")
    print(f"hakim: {HAKIM}")

    ds = Dataset.from_dict({
        "user_input": [d["user_input"] for d in data],
        "response": [d["response"] for d in data],
        "retrieved_contexts": [d["retrieved_contexts"] for d in data],
        "reference": [d["reference"] for d in data],
    })

    judge = LangchainLLMWrapper(ChatOllama(model=HAKIM, temperature=0))
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    metrik = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=emb),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    print("menilai... (lama)")
    hasil = evaluate(dataset=ds, metrics=metrik, llm=judge, embeddings=emb,
                     raise_exceptions=False)

    print()
    print("=" * 58)
    print(f"RAGAS — hakim: {HAKIM}")
    print("=" * 58)
    skor = dict(hasil._repr_dict)
    for k, v in skor.items():
        print(f"  {k:32s} = {v:.4f}")

    OUT.write_text(json.dumps({"hakim": HAKIM, **skor}, indent=2),
                   encoding="utf-8")
    hasil.to_pandas().to_csv(f"eval/ragas_detail_{HAKIM.replace(':','_')}.csv",
                             index=False)
    print(f"\ndisimpan -> {OUT}")