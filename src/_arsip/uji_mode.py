"""Bandingkan dense vs hybrid, 3 jalan masing-masing."""
import json
import re
import subprocess
from pathlib import Path

FILE = Path("../shared/retrieval.py")

for mode in ["dense", "hybrid"]:
    # ganti MODE di shared/retrieval.py
    isi = FILE.read_text(encoding="utf-8")
    isi = re.sub(r'^MODE = "\w+"', f'MODE = "{mode}"', isi, count=1, flags=re.M)
    FILE.write_text(isi, encoding="utf-8")

    print(f"\n=== {mode} ===")
    for i in range(3):
        subprocess.run(["python", "src/evaluate_jawaban.py"],
                       capture_output=True)
        d = json.loads(Path("eval/hasil_jawaban.json").read_text())
        print(f"  jalan {i+1}: akurasi {d['akurasi']:.3f} | "
              f"halusinasi {d['halusinasi']:.3f}")