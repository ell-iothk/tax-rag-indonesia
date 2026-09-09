#!/usr/bin/env bash
# Nyalakan semua layanan lalu masuk CLI RAG.
set -e

# folder tempat skrip ini berada, apa pun lokasinya
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CD_INFRA="$PROJ"

ok()   { echo "  [OK]   $1"; }
gagal(){ echo "  [GAGAL] $1"; exit 1; }

echo "=============================================="
echo "  Asisten Pajak — startup"
echo "=============================================="

# --- 1. Docker ---
echo "[1/4] layanan Docker"
if ! docker info >/dev/null 2>&1; then
    gagal "Docker tidak jalan. Jalankan: sudo service docker start"
fi
cd "$CD_INFRA"
docker compose start >/dev/null 2>&1 || docker compose up -d >/dev/null
sleep 3
ok "docker compose aktif"

# --- 2. Qdrant ---
echo "[2/4] Qdrant"
JML=$(curl -s http://localhost:6333/collections/tax_docs \
      | grep -o '"points_count":[0-9]*' | cut -d: -f2)
[ -z "$JML" ] && gagal "collection 'tax_docs' tidak ada. Jalankan: python scripts/index_qdrant.py"
ok "collection tax_docs: $JML titik"

# --- 3. Ollama ---
echo "[3/4] Ollama"
ollama list 2>/dev/null | grep -q qwen3-id || gagal "model qwen3-id tidak ada"
ok "model qwen3-id tersedia"

# --- 4. Python ---
echo "[4/4] Python environment"
if [ -d "$PROJ/.venv" ]; then
    source "$PROJ/.venv/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "  (memakai venv yang sudah aktif)"
else
    gagal "venv tidak ada. Jalankan: python3 -m venv .venv && pip install -e ."
fi

echo ""
cd "$PROJ"
python scripts/tanya.py
