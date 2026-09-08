"""Uji koneksi ke Langfuse dan kirim satu trace percobaan."""
import os
from dotenv import load_dotenv

# baca .env dari root project
load_dotenv(os.path.expanduser("~/ai-projects/.env"))

print("host  :", os.getenv("LANGFUSE_HOST"))
print("public:", (os.getenv("LANGFUSE_PUBLIC_KEY") or "")[:12], "...")

from langfuse import Langfuse

lf = Langfuse()

print("terhubung?", lf.auth_check())

# kirim satu trace percobaan
with lf.start_as_current_observation(name="uji-koneksi") as span:
    span.update(
        input={"pertanyaan": "tes dari WSL"},
        output={"jawaban": "berhasil"},
        metadata={"tahap": "step-2b"},
    )

lf.flush()
print("trace terkirim. buka http://localhost:3000 -> Tracing")