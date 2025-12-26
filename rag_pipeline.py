# rag_pipeline.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import time

from dotenv import load_dotenv
from datasets import load_dataset

# LangChain (your installed versions)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# NOTE: This triggers a deprecation warning in some versions.
# You can later switch to: from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

import rag_config as cfg

load_dotenv()


# ---------------- LLMs via Groq ---------------- #

def get_answer_llm() -> ChatGroq:
    return ChatGroq(
        model=cfg.ANSWER_MODEL_ID,
        temperature=cfg.ANSWER_TEMPERATURE,
        max_tokens=512,
        api_key=cfg.GROQ_API_KEY or None,
    )


def get_judge_llm() -> ChatGroq:
    return ChatGroq(
        model=cfg.JUDGE_MODEL_ID,
        temperature=cfg.JUDGE_TEMPERATURE,
        max_tokens=1024,
        api_key=cfg.GROQ_API_KEY or None,
    )


# ---------------- Embeddings ---------------- #

def get_embeddings(domain: str = "Finance") -> HuggingFaceEmbeddings:
    model_name = cfg.FINANCE_EMBEDDING_MODEL_ID if domain.lower() == "finance" else cfg.EMBEDDING_MODEL_ID
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------- Chunking (Sliding Window) ---------------- #

def split_documents_sliding(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# ---------------- RAGBench loading + caching ---------------- #

def _examples_cache_path() -> Path:
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.ARTIFACTS_DIR / f"ragbench_examples_{cfg.RAGBENCH_SUBSET}_{cfg.RAGBENCH_SPLIT}.json"


def load_ragbench_examples_cached(
    subset: str = cfg.RAGBENCH_SUBSET,
    split: str = cfg.RAGBENCH_SPLIT,
    max_rows: int = cfg.RAGBENCH_MAX_ROWS,
) -> List[Dict[str, Any]]:
    """
    Loads RAGBench examples and caches them to disk.
    Next runs load from artifacts JSON (fast, no HF delay).
    """
    cache_path = _examples_cache_path()
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                examples = json.load(f)
            # Ensure max_rows honored
            if max_rows and max_rows > 0:
                return examples[:max_rows]
            return examples
        except Exception:
            # If cache is corrupt, ignore and rebuild it.
            pass

    # Slow path: fetch from HF once
    ds = load_dataset(cfg.RAGBENCH_DATASET_ID, subset, split=split)
    n = min(len(ds), max_rows) if max_rows and max_rows > 0 else len(ds)
    examples = [ds[i] for i in range(n)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False)

    return examples


def _stringify_ragbench_doc(doc_obj: Any) -> str:
    """
    Robust conversion of RAGBench doc structures into plain text.
    """
    if doc_obj is None:
        return ""
    if isinstance(doc_obj, str):
        return doc_obj.strip()

    if isinstance(doc_obj, list):
        # list of strings
        if all(isinstance(x, str) for x in doc_obj):
            return "\n".join([x.strip() for x in doc_obj if x.strip()])

        lines = []
        for x in doc_obj:
            if isinstance(x, (list, tuple)):
                if len(x) == 2 and all(isinstance(y, str) for y in x):
                    lines.append(x[1].strip())
                else:
                    lines.append(_stringify_ragbench_doc(x))
            elif isinstance(x, dict):
                lines.append(json.dumps(x, ensure_ascii=False))
            else:
                lines.append(str(x))
        return "\n".join([l for l in lines if l.strip()])

    if isinstance(doc_obj, dict):
        for k in ["text", "passage", "content", "document"]:
            if k in doc_obj and isinstance(doc_obj[k], str):
                return doc_obj[k].strip()
        return json.dumps(doc_obj, ensure_ascii=False)

    return str(doc_obj).strip()


def ragbench_to_documents(examples: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for ex_i, ex in enumerate(examples):
        raw_docs = ex.get("documents") or ex.get("context") or ex.get("retrieved_docs") or []
        if raw_docs is None:
            raw_docs = []
        if not isinstance(raw_docs, list):
            raw_docs = [raw_docs]

        for doc_i, d in enumerate(raw_docs):
            text = _stringify_ragbench_doc(d)
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "ragbench",
                        "subset": cfg.RAGBENCH_SUBSET,
                        "split": cfg.RAGBENCH_SPLIT,
                        "example_index": ex_i,
                        "doc_index": doc_i,
                    },
                )
            )
    return docs


# ---------------- Fixed FAISS index path (NO cache misses) ---------------- #

def _fixed_index_dir(domain: str = "Finance") -> Path:
    """
    Stable index folder:
      artifacts/faiss_index/faiss_ragbench_finqa_test/
    """
    cfg.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.FAISS_INDEX_DIR / f"faiss_ragbench_{cfg.RAGBENCH_SUBSET}_{cfg.RAGBENCH_SPLIT}"


def load_faiss_index(domain: str = "Finance") -> Optional[FAISS]:
    embeddings = get_embeddings(domain)
    index_dir = _fixed_index_dir(domain)
    if not index_dir.exists():
        return None
    # must contain index.faiss + index.pkl
    if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
        return None
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def build_faiss_index_from_ragbench(domain: str = "Finance") -> Tuple[FAISS, List[Document], List[Dict[str, Any]]]:
    embeddings = get_embeddings(domain)

    examples = load_ragbench_examples_cached(
        subset=cfg.RAGBENCH_SUBSET,
        split=cfg.RAGBENCH_SPLIT,
        max_rows=cfg.RAGBENCH_MAX_ROWS,
    )

    base_docs = ragbench_to_documents(examples)
    chunks = split_documents_sliding(base_docs)

    print(f"[INDEX] Building FAISS over {len(chunks)} chunks from {cfg.RAGBENCH_SUBSET}/{cfg.RAGBENCH_SPLIT}...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir = _fixed_index_dir(domain)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"[INDEX] Saved to: {index_dir}")

    return vectorstore, chunks, examples


# ---------------- HyDE ---------------- #

def generate_hypothetical_answer(llm: ChatGroq, question: str) -> str:
    prompt = (
        "Write a detailed hypothetical answer to the question. "
        "This will be used only for retrieving documents.\n\n"
        f"Question:\n{question}\n\n"
        "Hypothetical answer:"
    )
    resp = llm.invoke(prompt)
    return resp.content.strip()


# ---------------- Reranker (lazy-load) ---------------- #

class BgeCrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        if not docs_with_scores:
            return []
        pairs = [(query, d.page_content) for d, _ in docs_with_scores]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs_with_scores, scores), key=lambda x: float(x[1]), reverse=True)
        return [(doc, float(score)) for (doc, _old), score in ranked[:top_k]]


# ---------------- Hybrid fusion ---------------- #

def _normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {k: 0.0 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}


def hybrid_fuse(
    dense: List[Tuple[Document, float]],
    sparse: List[Document],
    dense_weight: float,
    top_k: int,
) -> List[Tuple[Document, float]]:
    dense_scores = {d.page_content: float(s) for d, s in dense}
    bm25_scores = {d.page_content: 1.0 / (i + 1) for i, d in enumerate(sparse)}

    dense_norm = _normalize_scores(dense_scores)
    bm25_norm = _normalize_scores(bm25_scores)

    keys = set(dense_scores.keys()) | set(bm25_scores.keys())

    doc_lookup: Dict[str, Document] = {}
    for d, _ in dense:
        doc_lookup[d.page_content] = d
    for d in sparse:
        doc_lookup.setdefault(d.page_content, d)

    fused: List[Tuple[Document, float]] = []
    for k in keys:
        score = dense_weight * dense_norm.get(k, 0.0) + (1.0 - dense_weight) * bm25_norm.get(k, 0.0)
        fused.append((doc_lookup[k], float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


# ---------------- Repacking (Reverse) ---------------- #

def repack_context_reverse(docs_with_scores: List[Tuple[Document, float]], max_chars: int = 8000) -> str:
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    sorted_docs = list(reversed(sorted_docs))  # reverse repack (best at end)

    parts: List[str] = []
    total = 0
    for doc, _score in sorted_docs:
        t = doc.page_content.strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            break
        parts.append(t)
        total += len(t)

    return "\n\n---\n\n".join(parts)


# ---------------- Summarisation (Recomp-style) ---------------- #

def summarize_context_recomp(llm: ChatGroq, question: str, context: str) -> str:
    prompt = (
        "Given QUESTION and CONTEXT, write a concise factual abstractive summary "
        "keeping ONLY information needed to answer the question.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nSUMMARY:"
    )
    resp = llm.invoke(prompt)
    return resp.content.strip()


# ---------------- Answer generation ---------------- #

def build_answer_prompt(question: str, context: str) -> str:
    return (
        "You are a highly accurate RAG assistant. Answer using ONLY the context. "
        "If not present, say exactly: \"I don't know based on the provided context.\".\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
    )


def ask_answer_model(llm: ChatGroq, question: str, context: str) -> str:
    resp = llm.invoke(build_answer_prompt(question, context))
    return resp.content.strip()


# ---------------- TRACe Judge (robust JSON parsing) ---------------- #

def judge_answer_trace(judge_llm: ChatGroq, question: str, context: str, answer: str) -> Dict[str, Any]:
    prompt = f"""
Return ONLY valid JSON. No markdown, no code fences, no extra text.

Schema:
{{
  "relevance": float,
  "utilization": float,
  "completeness": float,
  "adherence": bool,
  "relevant_sentence_indices": [int],
  "utilized_sentence_indices": [int],
  "unsupported_answer_sentence_indices": [int],
  "explanation": str
}}

Rules:
- relevance/utilization/completeness must be numbers between 0 and 1 (round to 2 decimals).
- adherence must be true/false.
- If there is not enough evidence in CONTEXT, set adherence=false.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

JSON:
""".strip()

    resp = judge_llm.invoke(prompt)
    text = resp.content.strip()

    # 1) direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2) extract substring {...}
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    return {
        "relevance": None,
        "utilization": None,
        "completeness": None,
        "adherence": None,
        "relevant_sentence_indices": [],
        "utilized_sentence_indices": [],
        "unsupported_answer_sentence_indices": [],
        "explanation": text[:1200],
    }


# ---------------- RagEngine ---------------- #

class RagEngine:
    """
    RAG engine backed by RAGBench subset/split.
    Loads FAISS from fixed artifacts path to avoid rebuild.
    Caches dataset examples to artifacts JSON for fast startup.
    Lazy-loads reranker so app UI starts quickly.
    """

    def __init__(
        self,
        domain: str = "Finance",
        use_hyde: bool = cfg.USE_HYDE_DEFAULT,
        use_rerank: bool = cfg.USE_RERANK_DEFAULT,
        use_summarisation: bool = cfg.USE_SUMMARIZATION_DEFAULT,
    ):
        self.domain = domain
        self.use_hyde = use_hyde
        self.use_rerank = use_rerank
        self.use_summarisation = use_summarisation

        t0 = time.time()
        print("[INIT] RagEngine init...")

        self.answer_llm = get_answer_llm()
        self.judge_llm = get_judge_llm()

        # Lazy-load examples only when row-based queries are used.
        self._examples: Optional[List[Dict[str, Any]]] = None

        # Load FAISS (fast) or build once (slow)
        print(f"[INIT] Loading FAISS index from: {_fixed_index_dir(domain)}")
        vs = load_faiss_index(domain)
        if vs is None:
            print("[INIT] FAISS index not found -> building (first time slow)")
            vs, _chunks, _ex = build_faiss_index_from_ragbench(domain)
        else:
            print("[INIT] FAISS loaded from disk ✅")

        self.vectorstore: FAISS = vs

        # BM25 over FAISS docstore chunks
        docs = list(self.vectorstore.docstore._dict.values())
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = cfg.TOP_K * 2

        # Reranker lazy-load flags
        self.reranker: Optional[BgeCrossEncoderReranker] = None
        self._reranker_loaded = False

        self.dense_k = cfg.TOP_K * 2

        print(f"[INIT] RagEngine ready in {time.time() - t0:.2f}s")

    def _ensure_examples_loaded(self) -> None:
        if self._examples is None:
            self._examples = load_ragbench_examples_cached(
                subset=cfg.RAGBENCH_SUBSET,
                split=cfg.RAGBENCH_SPLIT,
                max_rows=cfg.RAGBENCH_MAX_ROWS,
            )
            print(f"[INIT] Examples loaded: {len(self._examples)}")

    def get_example(self, row_index: int) -> Dict[str, Any]:
        self._ensure_examples_loaded()
        assert self._examples is not None
        if row_index < 0 or row_index >= len(self._examples):
            raise IndexError(f"row_index out of range: 0..{len(self._examples)-1}")
        return self._examples[row_index]

    def _extract_question(self, ex: Dict[str, Any]) -> str:
        for k in ["question", "query", "instruction", "prompt"]:
            if k in ex and isinstance(ex[k], str) and ex[k].strip():
                return ex[k].strip()
        return str(ex)

    def _extract_ground_truth(self, ex: Dict[str, Any]) -> Optional[str]:
        for k in ["answer", "ground_truth", "response", "reference_answer"]:
            if k in ex and isinstance(ex[k], str) and ex[k].strip():
                return ex[k].strip()
        return None

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        # Dense (FAISS)
        if self.use_hyde:
            hyp = generate_hypothetical_answer(self.answer_llm, query)
            dense_results = self.vectorstore.similarity_search_with_score(hyp, k=self.dense_k)
        else:
            dense_results = self.vectorstore.similarity_search_with_score(query, k=self.dense_k)

        dense = [(d, float(s)) for d, s in dense_results]

        # Sparse (BM25) - use invoke() for compatibility
        sparse_docs = self.bm25_retriever.invoke(query)
        if not isinstance(sparse_docs, list):
            sparse_docs = list(sparse_docs)

        # Fuse
        fused = hybrid_fuse(
            dense=dense,
            sparse=sparse_docs,
            dense_weight=cfg.HYBRID_DENSE_WEIGHT,
            top_k=cfg.TOP_K * 2,
        )

        # Optional rerank (lazy-load)
        if self.use_rerank:
            if not self._reranker_loaded:
                print("[RERANK] Loading cross-encoder reranker (first use may take time)...")
                self.reranker = BgeCrossEncoderReranker()
                self._reranker_loaded = True

            if self.reranker is not None:
                fused = self.reranker.rerank(query, fused, top_k=cfg.TOP_K)
            else:
                fused = fused[: cfg.TOP_K]
        else:
            fused = fused[: cfg.TOP_K]

        return fused

    def answer(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query)

        context_text = repack_context_reverse(retrieved, max_chars=8000)

        if self.use_summarisation and len(context_text) > 4000:
            compressed_context = summarize_context_recomp(self.answer_llm, query, context_text)
        else:
            compressed_context = context_text

        answer = ask_answer_model(self.answer_llm, query, compressed_context)
        judge = judge_answer_trace(self.judge_llm, query, compressed_context, answer)

        return {
            "question": query,
            "answer": answer,
            "context": compressed_context,
            "retrieved": retrieved,
            "judge": judge,
        }

    def answer_from_row(self, row_index: int) -> Dict[str, Any]:
        ex = self.get_example(row_index)
        q = self._extract_question(ex)
        gt = self._extract_ground_truth(ex)

        out = self.answer(q)
        out["row_index"] = row_index
        out["ground_truth"] = gt
        return out
