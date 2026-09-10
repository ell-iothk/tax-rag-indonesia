"""CLI interaktif untuk RAG perpajakan."""

import time

from taxrag.models import get_tracer
from taxrag.rag import buat_chain
from taxrag.retrieval import MODE, TOP_N

BANNER = f"""
{"=" * 62}
  ASISTEN PAJAK — RAG lokal
  korpus  : PMK 168/2023, PP 58/2023
  mode    : {MODE}, top_n={TOP_N}
{"=" * 62}
  ketik pertanyaan, atau:
    /sumber   tampilkan chunk yang dipakai jawaban terakhir
    /keluar   selesai
"""


def main():
    print(BANNER)
    print("memuat model...")
    chain = buat_chain()
    tracer = get_tracer()
    terakhir = None
    print("siap.\n")

    while True:
        try:
            q = input("tanya> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nselesai.")
            break

        if not q:
            continue
        if q in ("/keluar", "/exit", "/q"):
            print("selesai.")
            break
        if q == "/sumber":
            if not terakhir:
                print("  belum ada pertanyaan.\n")
                continue
            for i, d in enumerate(terakhir["docs"], 1):
                m = d.metadata
                label = m.get("pasal") or m.get("bab") or "?"
                print(f"\n  [{i}] {m.get('doc')} | {label}")
                print(f"      {d.page_content[:200].replace(chr(10), ' ')}")
            print()
            continue

        t0 = time.perf_counter()
        try:
            terakhir = chain.invoke({"pertanyaan": q}, config={"callbacks": [tracer]})
        except Exception as e:
            print(f"  error: {e}\n")
            continue
        dt = time.perf_counter() - t0

        print(f"\n{terakhir['jawaban'].strip()}")
        sumber = []
        for d in terakhir["docs"]:
            m = d.metadata
            s = f"{m.get('doc')} {m.get('pasal') or m.get('bab') or ''}".strip()
            if s not in sumber:
                sumber.append(s)
        print(f"\n  sumber ({dt:.1f}s): {' | '.join(sumber[:3])}")
        print("  /sumber untuk lihat isi lengkapnya\n")


if __name__ == "__main__":
    main()
