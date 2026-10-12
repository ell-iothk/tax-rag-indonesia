.PHONY: help setup install models llm up up-obs down index chat eval eval-jawaban latensi lint api api-up image

PYTHON := .venv/bin/python

api: ## Jalankan API lokal di :8000
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

api-up: ## Nyalakan API lewat Docker
	docker compose --profile api up -d --build

image: ## Build image API
	docker build -t ghcr.io/ell-iothk/tax-rag-indonesia:latest .

help: ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ---------- setup (sekali saja) ----------

setup: install models llm .env ## Pasang semuanya: venv, model, LLM, .env

install: ## Buat venv dan pasang paket
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

models: ## Unduh bge-m3 dan reranker (~4,4 GB)
	$(PYTHON) scripts/download_models.py

llm: ## Siapkan model Qwen3 di Ollama
	ollama pull qwen3:4b-instruct-2507-q4_K_M
	ollama create qwen3-id -f Modelfile.qwen3-id

.env:
	cp .env.example .env

# ---------- layanan ----------

up: ## Nyalakan Qdrant
	docker compose up -d

up-obs: ## Nyalakan Qdrant + Langfuse
	docker compose --profile observability up -d

down: ## Matikan semua layanan
	docker compose --profile observability down

# ---------- pakai ----------

index: up ## Parsing PDF lalu isi Qdrant
	$(PYTHON) scripts/ingest.py
	$(PYTHON) scripts/index_qdrant.py

chat: ## Masuk CLI tanya-jawab
	./rag.sh

# ---------- evaluasi ----------

eval: ## Metrik retrieval
	$(PYTHON) scripts/evaluate.py

eval-jawaban: ## Metrik kualitas jawaban (butuh Ollama)
	$(PYTHON) scripts/evaluate_jawaban.py

latensi: ## Ukur latensi p50 dan p95
	$(PYTHON) scripts/ukur_latensi.py

# ---------- kualitas kode ----------

lint: ## Cek gaya kode dengan ruff
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
