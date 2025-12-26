import streamlit as st
from dotenv import load_dotenv

import rag_config as cfg
from rag_pipeline import RagEngine

load_dotenv()

st.set_page_config(page_title="RAGBench RAG — FinQA", layout="wide")
st.title("📊 RAGBench RAG — FinQA")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Accuracy (Hybrid + HyDE + Rerank + Summarise)", "Fast (Hybrid only)"],
    index=0,
)

if mode.startswith("Accuracy"):
    use_hyde = True
    use_rerank = True
    use_summarisation = True
else:
    use_hyde = False
    use_rerank = False
    use_summarisation = False

st.sidebar.caption(f"Dataset: {cfg.RAGBENCH_DATASET_ID}")
st.sidebar.caption(f"Subset: {cfg.RAGBENCH_SUBSET} | Split: {cfg.RAGBENCH_SPLIT}")
st.sidebar.caption(f"Max rows indexed: {cfg.RAGBENCH_MAX_ROWS}")

# ---------------- Mode selector: Row vs Free text ----------------
input_mode = st.radio(
    "Choose how you want to query:",
    ["Use RAGBench row index (reproducible)", "Type your own question (free-text)"],
    horizontal=True,
)

# Build engine on demand (cache across runs)
@st.cache_resource(show_spinner=False)
def get_engine(use_hyde: bool, use_rerank: bool, use_summarisation: bool):
    return RagEngine(
        domain="Finance",
        use_hyde=use_hyde,
        use_rerank=use_rerank,
        use_summarisation=use_summarisation,
    )

# ---------------- Inputs ----------------
row_index = None
free_question = None

if input_mode.startswith("Use RAGBench row index"):
    row_index = st.number_input(
        "Enter RAGBench row index (e.g., 2):",
        min_value=0,
        value=2,
        step=1,
    )
else:
    free_question = st.text_area(
        "Enter your finance question (try paraphrases / different wording):",
        height=120,
        placeholder="Example: How much of the event-driven loans and commitments will mature in 2014 (in billions)?",
    )
    st.caption("Tip: try rephrasing the same intent in 2–3 different ways to test retrieval robustness.")

run = st.button("Run RAG", type="primary")

# ---------------- Run ----------------
if run:
    engine = get_engine(use_hyde, use_rerank, use_summarisation)
    with st.spinner("Retrieving, answering, judging..."):
        try:
            if row_index is not None:
                result = engine.answer_from_row(int(row_index))
            else:
                if not free_question or not free_question.strip():
                    st.warning("Please type a question first.")
                    st.stop()
                result = engine.answer(free_question.strip())
                result["row_index"] = None
                result["ground_truth"] = None

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ---------------- Display ----------------
    st.subheader("🧾 Question")
    st.write(result["question"])

    if result.get("ground_truth"):
        with st.expander("✅ Ground truth (if present in dataset)"):
            st.write(result["ground_truth"])

    st.subheader("🧠 Model Answer")
    st.write(result["answer"])

    st.subheader("📊 TRACe Judge Metrics")
    j = result.get("judge", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevance", j.get("relevance"))
    c2.metric("Utilization", j.get("utilization"))
    c3.metric("Completeness", j.get("completeness"))
    adh = j.get("adherence")
    c4.metric("Adherence", "✅ True" if adh is True else "❌ False" if adh is False else "N/A")

    with st.expander("📚 Context used"):
        st.write(result["context"])

    with st.expander("🔎 Retrieved chunks"):
        for i, (doc, score) in enumerate(result["retrieved"], 1):
            st.markdown(f"**Chunk {i} (score={score:.4f})**")
            st.write(doc.page_content)
            st.caption(str(doc.metadata))
            st.markdown("---")

    with st.expander("🧪 Debug: Judge raw JSON"):
        st.json(result.get("judge", {}))
