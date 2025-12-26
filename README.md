# realworld_rag_finance

Compact example of a Retrieval-Augmented Generation (RAG) pipeline for finance question answering using FAISS indexes.

## Overview

This repository provides a minimal pipeline for indexing finance data, retrieving relevant documents with FAISS, and producing answers using a generative model. It includes configuration (`rag_config.py`), pipeline logic (`rag_pipeline.py`) and an example entrypoint (`app.py`).

## Prerequisites

- Python 3.8+ (3.10+ recommended)
- git
- Optional: CUDA + GPU-supporting libraries if you run heavy models locally

Use a virtual environment to isolate dependencies. A local venv exists at `.venv` in this workspace but you may prefer creating your own.

## Setup

Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
# Optionally install extras
pip install -r requirements-optional.txt || true
```

Environment variables (examples)

```bash
# Path to FAISS index directory
export RAG_FAISS_DIR=artifacts/faiss_index/faiss_ragbench_finqa_test
# Optional model config (depends on your pipeline)
export RAG_MODEL_NAME="local-or-remote-model-id"
```

## Quick start

1. Verify there is at least one FAISS index under `artifacts/faiss_index/` (this repo includes example indexes).
2. Edit `rag_config.py` if you need to point to a different index or model.
3. Run the example app:

```bash
python app.py
```

What to expect: `app.py` will load configuration from `rag_config.py`, initialize the retrieval index from the `artifacts/` folder, then run the pipeline defined in `rag_pipeline.py`. See those files for the precise behavior and supported CLI/options.

## Rebuilding FAISS indexes (summary)

If you want to regenerate FAISS indexes from source data in `data/finance/`, a typical flow is:

```bash
# 1. Prepare or preprocess source documents (tokenize, split into chunks)
# 2. Create embeddings for each chunk (using your embedding model)
# 3. Build a FAISS index and save it under `artifacts/faiss_index/<index_name>/`
# Example (pseudo):
python tools/build_index.py --data-dir data/finance --out-dir artifacts/faiss_index/my_index
```

There is no universal build script in this repo; inspect `rag_pipeline.py` for any helper functions and I can add a dedicated `tools/build_index.py` when you want it.

## Configuration details

- `rag_config.py`: model selection, index paths, retrieval parameters (k for top-k), and other runtime options.
- `rag_pipeline.py`: shows how retrieval and generation are composed. Modify it to change retrieval scoring, chunking, or how results are combined.

## Troubleshooting

- If the index doesn't load, confirm the `index.faiss` file exists under the target `artifacts/faiss_index/<dir>/` and that `rag_config.py` points to the correct path.
- If embeddings mismatch, ensure the embedding model used when building the index matches the one used to query it.
- If model inference is slow, consider using smaller models or GPU-accelerated runtimes.

## Project layout

- `app.py` — example entrypoint.
- `rag_config.py` — configuration for models and paths.
- `rag_pipeline.py` — retrieval + generation pipeline implementation.
- `requirements.txt` — required packages.
- `requirements-optional.txt` — optional packages.
- `artifacts/` — bundled/generated artifacts (FAISS indexes, example data outputs).
  - `artifacts/faiss_index/` — prebuilt FAISS index directories.
- `data/finance/` — source data for building indexes.

## Contributing / Next steps

- To run locally: activate venv, install deps, and run `python app.py`.
- To add reproducibility: I can add a `tools/build_index.py` script that converts `data/finance/` into a FAISS index using a chosen embedding model.
- To add examples: I can include sample queries and expected responses for smoke tests.

## License

This repository now includes an `LICENSE` file (MIT). See `LICENSE` for details.
