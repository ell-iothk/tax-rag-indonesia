"""Cek jawaban golden set ada di PDF, pakai pencocokan token."""
import json
import re
import subprocess
from pathlib import Path

RAW = Path("data/raw")

teks = []
for pdf in sorted(RAW.glob("*.pdf")):
    hasil = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                           capture_output=True, text=True)
    teks.append(hasil.stdout)
dok = " ".join(" ".join(teks).split()).lower()

# kata umum yang tidak membedakan apa-apa
STOP = {"dan", "atau", "yang", "untuk", "dari", "dengan", "pada", "di",
        "ke", "dalam", "adalah", "itu", "ini", "sebesar", "paling", "bisa",
        "tidak", "oleh", "sesuai", "karena", "saja", "beserta", "juga"}


def token_khas(teks: str) -> list[str]:
    """Ambil kata yang cukup khas: angka, atau kata >4 huruf."""
    kata = re.findall(r"[a-zA-Z]{5,}|[\d.]+%?|rp[\d.]+", teks.lower())
    return [k for k in kata if k not in STOP]


lolos, gagal = 0, []
for line in open("eval/golden_set.jsonl", encoding="utf-8"):
    q = json.loads(line)
    if q["category"] == "unanswerable":
        continue
    tok = token_khas(q["answer"])
    if not tok:
        continue
    ada = [t for t in tok if t in dok]
    rasio = len(ada) / len(tok)
    if rasio >= 0.7:
        lolos += 1
    else:
        gagal.append((q["id"], round(rasio, 2), [t for t in tok if t not in dok]))

print(f"lolos : {lolos}")
print(f"gagal : {len(gagal)}")
for i, r, hilang in gagal:
    print(f"  {i} (cocok {r:.0%}) kata tak ada: {hilang}")