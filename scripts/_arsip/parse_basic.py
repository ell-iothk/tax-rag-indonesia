"""Ekstrak teks dari PDF per halaman."""
import json
import pdfplumber
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")


def parse_pdf(pdf_path: Path) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            pages.append({"doc": pdf_path.stem, "page": i, "text": text})
    return pages


if __name__ == "__main__":
    berkas = sorted(RAW.glob("*.pdf"))
    if not berkas:
        raise SystemExit(f"Tidak ada PDF di {RAW}")

    for pdf in berkas:
        pages = parse_pdf(pdf)
        out_file = OUT / f"{pdf.stem}_pages.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        total = sum(len(p["text"]) for p in pages)
        kosong = sum(1 for p in pages if len(p["text"].strip()) < 50)
        print(f"{pdf.name}: {len(pages)} hal | {total:,} karakter | {kosong} hal kosong")