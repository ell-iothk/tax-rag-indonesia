import json

for line in open("data/processed/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    if c.get("pasal") == "Pasal 3" and "pengacara" in c["text"].lower():
        print("panjang:", len(c["text"]), "karakter")
        print()
        i = c["text"].lower().find("tenaga ahli")
        print("posisi 'tenaga ahli':", i, "dari", len(c["text"]))
        print()
        print("--- sekitar kata itu ---")
        print(c["text"][max(0, i - 250): i + 350])
        break