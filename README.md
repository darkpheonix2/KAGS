# KAGS — Knowledge graph Augmented Generation System

Implementation and experiments for **Knowledge graph Augmented Generation System (KAGS)** and baseline RAG methods on multi-hop and open-domain QA benchmarks.

Repository: [github.com/darkpheonix2/KAGS](https://github.com/darkpheonix2/KAGS)

## Overview

KAGS combines **knowledge-graph construction**, **hybrid vector + graph retrieval**, and **LLM answer generation** to answer complex questions. The codebase also includes baselines (Vector RAG, Hybrid RAG, Adaptive RAG, Graph RAG) and evaluation scripts across four datasets.

```
Question → KGC (triplets) → KGE (Weaviate + Neo4j) → KGR (retrieve context) → KGA (generate answer) → Metrics
```

## Notebooks

Jupyter notebooks (`.ipynb`) are **git-ignored** because they may contain old API keys. Keep them on your machine only; use the `.py` scripts for reproducible runs.

## Repository layout

| Path | Description |
|------|-------------|
| `HotpotQA/` | HotpotQA experiments (multi-hop Wikipedia QA) |
| `2wikimultihopqa/` | 2WikiMultihopQA experiments |
| `NaturalQA/` | Natural Questions experiments |
| `TriviaQA/` | TriviaQA experiments |

Each dataset folder is self-contained and follows the same module naming convention.

## Core modules (per dataset)

| File | Role |
|------|------|
| `kgc.py` | **Knowledge graph construction** — chunk context, extract `(subject, predicate, object, metadata)` triplets with an LLM |
| `kge.py` | **Knowledge graph embedding & storage** — embed triplets, ingest into Weaviate (vector) and Neo4j (graph) |
| `kgr.py` | **Knowledge graph retrieval** — vector search + metadata filtering + multi-hop graph traversal |
| `kga.py` | **Knowledge augmented generation** — answer generation from retrieved context |
| `metrics.py` | Evaluation metrics (e.g. faithfulness, answer similarity) |
| `Evaluating*.py` | Batch evaluation for Flan-T5, Llama, Mistral, etc. |

### Baselines & variants

| Pattern | Description |
|---------|-------------|
| `Vector_RAG*.py` | Dense retrieval + generation (no graph) |
| `Hybrid_RAG*.py` / `kga_HybridRAG.py` | Combined vector and graph signals |
| `Adaptive_RAG*.py` | Adaptive routing between strategies |
| `*_GraphRAG.py` | Graph-focused RAG pipeline |
| `Without Retrieval*.py` | LLM-only (no retrieval) |
| `kgr_h_*.py`, `kgr_k.py` | Ablation scripts (hop count, top-k, thresholds) |
| `AblationStudy/` | Ablation run outputs (CSV) |
| `*.ipynb` | Interactive notebooks for pipeline steps and analysis |

## Prerequisites

- **Python** 3.10+
- **CUDA** GPU(s) recommended for embedding models and LLMs
- **Weaviate Cloud** (or self-hosted) for vector storage
- **Neo4j** (Aura or self-hosted) for graph storage
- **Hugging Face** access for downloaded models

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r HotpotQA/requirements.txt
```

Additional packages used in some scripts: `weaviate-client`, `neo4j`, `nltk` (run `python -m nltk.downloader punkt punkt_tab` if needed).

## Configuration

Credentials are loaded from environment variables via `db_config.py` (uses `python-dotenv`).

1. Copy the example file:
   ```bash
   cd RAGs
   cp .env.example .env
   ```
2. Edit `.env` with your Weaviate, Neo4j, and Hugging Face values (never commit `.env`).
3. When switching datasets (HotpotQA vs TriviaQA use different cloud instances), update `.env` to match that dataset’s cluster before running.

| Variable | Purpose |
|----------|---------|
| `WEAVIATE_URL` | Weaviate cluster hostname |
| `WEAVIATE_API_KEY` | Weaviate API key |
| `NEO4J_URI` | Neo4j connection URI |
| `NEO4J_USER` | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |
| `HF_TOKEN` | Hugging Face access token |

Scripts import helpers such as `get_weaviate_client()`, `get_neo4j_driver()`, and `setup_hf_token()` from `db_config`.

**Security:** If credentials were ever committed to git, rotate them in Weaviate, Neo4j, and Hugging Face before pushing.

Update collection names in `kge.py` / `kgr.py` (e.g. `triplet_database_name`) for your project.

## Typical workflow

Run from the dataset directory you are working on (example: **HotpotQA**):

### 1. Build triplets (KGC)

```bash
cd HotpotQA
python kgc.py
```

Produces triplet CSVs from preprocessed QA context.

### 2. Embed and ingest (KGE)

```bash
python kge.py
```

Embeds triplets and writes them to Weaviate and Neo4j.

### 3. Retrieve context (KGR)

```bash
python kgr.py
```

Runs vector + graph retrieval; writes retrieval output (e.g. `Test_kgr.csv`).

### 4. Generate answers (KGA)

```bash
python kga.py
```

Reads retrieval output and writes predictions.

### 5. Evaluate

```bash
python Evaluating.py          # or Evaluating_llama.py / Evaluating_mistral.py / Evaluating_flan.py
python metrics.py             # metric utilities used by evaluation scripts
```

Notebooks such as `Evaluating.ipynb` and `KGE.ipynb` mirror these steps for interactive use.

## Models

Scripts support multiple backends (configure in each file’s `from_pretrained` block):

- **Phi-4-mini-instruct** (default in several HotpotQA scripts)
- **google/flan-t5-xl**
- **meta-llama/Llama-3.2-3B-Instruct**
- **mistralai/Mistral-7B-Instruct-v0.3**

Sentence embeddings typically use **sentence-transformers** (e.g. `all-MiniLM-L6-v2` or project-specific models).

## Large files & datasets

Some raw datasets and checkpoint files exceed GitHub’s file size limit and are **not tracked** in git (see `.gitignore`). Download or generate them locally:

| File (examples) | Dataset |
|-----------------|---------|
| `NaturalQA/biencoder-nq-dev.json` | Natural Questions |
| `TriviaQA/triviaqa-dev_new.json` | TriviaQA |
| Large `*checkpoint*.csv`, `GraphRAG_*.csv` | Generated during KGC/KGR runs |

Place subsampled test CSVs (e.g. `test_subsampled.csv`) in the appropriate folder as referenced by each script.


## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Open a pull request with a clear description of changes  

Issues and pull requests are welcome.
