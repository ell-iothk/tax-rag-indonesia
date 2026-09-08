"""RAG chain: retrieval -> prompt -> LLM -> jawaban."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from shared.models import get_llm, get_tracer
from shared.retrieval import get_retriever

COLLECTION = "tax_docs"
CHUNKS = "data/processed/chunks.jsonl"

TEMPLATE = """Kamu asisten pajak Indonesia. Jawab HANYA dari KONTEKS di bawah.

CARA MENJAWAB:
1. Baca SEMUA potongan konteks [1] sampai [5]. Jawaban bisa ada di potongan
   mana pun, bukan hanya yang pertama.
2. Cocokkan ISTILAH di pertanyaan dengan istilah di konteks secara TEPAT.
   Kalau pertanyaan menyebut suatu istilah, cari daftar/definisi untuk
   istilah ITU, bukan istilah lain yang berdekatan di pasal yang sama.
3. Kalau ada potongan yang memuat jawabannya, jawab dari situ dan sebutkan
   nomor potongan serta sumbernya. Contoh: (potongan [3], PMK 168/2023 Pasal 10)
4. Kalau BENAR-BENAR tidak ada satu pun potongan yang memuat jawabannya,
   jawab persis: "Tidak ditemukan dalam dokumen."

LARANGAN:
- DILARANG memakai pengetahuan di luar konteks, walaupun kamu tahu jawabannya.
- DILARANG menyimpulkan dari informasi yang tidak langsung menjawab pertanyaan.
- DILARANG menggabungkan "Tidak ditemukan dalam dokumen" dengan jawaban lain.
- Jawab maksimal 3 kalimat.

KONTEKS:
{konteks}

PERTANYAAN: {pertanyaan}

JAWABAN:"""


def format_konteks(docs) -> str:
    """Gabungkan chunk jadi satu teks, tiap chunk diberi label sumber."""
    bagian = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        sumber = f"{m.get('doc', '?')} {m.get('pasal') or m.get('bab') or ''}".strip()
        bagian.append(f"[{i}] Sumber: {sumber}\n{d.page_content}")
    return "\n\n".join(bagian)


def buat_chain(mode: str = None, top_n: int = 5):
    retriever = get_retriever(COLLECTION, CHUNKS,
                             **({"mode": mode} if mode else {}),
                             top_n=top_n)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_llm(temperature=0)

    # ambil dokumen sekali, pakai untuk konteks DAN dikembalikan ke pemanggil
    ambil = RunnableParallel(
        docs=lambda x: retriever.invoke(x["pertanyaan"]),
        pertanyaan=lambda x: x["pertanyaan"],
    )

    jawab = (
        RunnablePassthrough.assign(konteks=lambda x: format_konteks(x["docs"]))
        | RunnableParallel(
            jawaban=prompt | llm | StrOutputParser(),
            docs=lambda x: x["docs"],
        )
    )

    return ambil | jawab


if __name__ == "__main__":
    chain = buat_chain()
    tracer = get_tracer()

    PERTANYAAN = [
        "Biaya jabatan maksimal berapa setahun?",
        "Selebgram masuk kategori penerima penghasilan apa?",
        "Berapa gaji rata-rata konsultan pajak di Jakarta?",
    ]

    for q in PERTANYAAN:
        print("=" * 70)
        print("TANYA :", q)
        hasil = chain.invoke({"pertanyaan": q},
                             config={"callbacks": [tracer]})
        print("JAWAB :", hasil["jawaban"].strip())
        sumber = [f"{d.metadata.get('doc')} {d.metadata.get('pasal') or ''}".strip()
                  for d in hasil["docs"]]
        print("CHUNK :", sumber)
        print()