"""PDF -> chunk. Semua logika ada di taxrag/chunking.py."""

import json
from pathlib import Path

from taxrag.chunking import pipeline_standar

RAW = Path("data/raw")
OUT = Path("data/processed/chunks.jsonl")

if __name__ == "__main__":
    berkas = sorted(RAW.glob("*.pdf"))
    if not berkas:
        raise SystemExit(f"Tidak ada PDF di {RAW}")

    semua = []
    for pdf in berkas:
        docs = pipeline_standar(pdf)
        semua.extend(docs)
        print(f"{pdf.name}: {len(docs)} chunk")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, d in enumerate(semua, 1):
            f.write(
                json.dumps(
                    {
                        "id": f"c{i:04d}",
                        "text": d.page_content,
                        **d.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    p = [len(d.page_content) for d in semua]
    print(f"\nTOTAL {len(semua)} chunk -> {OUT}")
    print(f"karakter : min {min(p)} | rata {sum(p) // len(p)} | maks {max(p)}")
    print(f"token    : rata {sum(p) / len(p) / 3.5:.0f} | maks {max(p) / 3.5:.0f}")
    tanpa_pasal = sum(1 for d in semua if not d.metadata.get("pasal"))
    tanpa_apa2 = sum(1 for d in semua if not d.metadata.get("pasal") and not d.metadata.get("bab"))
    print(f"tanpa pasal      : {tanpa_pasal}")
    print(f"tanpa bab & pasal: {tanpa_apa2}   <- harus 0")
