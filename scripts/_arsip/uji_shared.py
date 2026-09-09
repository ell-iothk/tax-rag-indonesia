import os
from shared.models import get_embeddings, EMBED_MODEL

print("HF_HOME:", os.environ["HF_HOME"])
print("model  :", EMBED_MODEL)

a = get_embeddings()
b = get_embeddings()   # tidak boleh memuat ulang

print("objek sama?", a is b)
print("dimensi   :", len(a.embed_query("tes")))
