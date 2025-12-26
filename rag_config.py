# rag_config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FAISS_INDEX_DIR = ARTIFACTS_DIR / "faiss_index"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---- Groq ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set via environment; do NOT store secrets in source
ANSWER_MODEL_ID = os.getenv("ANSWER_MODEL_ID", "llama-3.3-70b-versatile")
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "llama-3.3-70b-versatile")

# ---- Embeddings ----
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/llm-embedder")
FINANCE_EMBEDDING_MODEL_ID = os.getenv("FINANCE_EMBEDDING_MODEL_ID", EMBEDDING_MODEL_ID)

# ---- RAGBench (HF dataset) ----
# Your link uses galileo-ai; keep that as default
RAGBENCH_DATASET_ID = os.getenv("RAGBENCH_DATASET_ID", "galileo-ai/ragbench")
RAGBENCH_SUBSET = os.getenv("RAGBENCH_SUBSET", "finqa")
RAGBENCH_SPLIT = os.getenv("RAGBENCH_SPLIT", "test")  # "train" / "validation" / "test"

# IMPORTANT: indexing full finqa_train may be heavy on CPU.
# Use a cap for local dev; you can raise later.
RAGBENCH_MAX_ROWS = int(os.getenv("RAGBENCH_MAX_ROWS", "2000"))

# ---- Chunking (sliding window) ----
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ---- Retrieval ----
TOP_K = 8
HYBRID_DENSE_WEIGHT = 0.7

# ---- Modes ----
USE_HYDE_DEFAULT = True
USE_RERANK_DEFAULT = True
USE_SUMMARIZATION_DEFAULT = True

# ---- Temperatures ----
JUDGE_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2

SUPPORTED_DOMAINS = ["Finance"]
