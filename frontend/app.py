"""
Streamlit Frontend — Advanced Cyclic RAG with LangGraph & Groq
A clean interface showing the full RAG pipeline with transparency into each step.
"""
import streamlit as st
import httpx

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Cyclic RAG — AI Knowledge Base",
    page_icon="🧠",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Advanced Cyclic RAG")
st.markdown("*Powered by LangGraph + Groq + ChromaDB — with self-correction loops*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ System Info")

try:
    r = httpx.get(f"{API_BASE}/docs-count", timeout=5)
    info = r.json()
    st.sidebar.metric("📚 Knowledge Base Docs", info["count"])
    st.sidebar.caption(f"Topic: {info['topic']}")
except Exception:
    st.sidebar.warning("Backend not connected")

st.sidebar.divider()
st.sidebar.markdown("### 🔄 How the Cyclic RAG Works")
st.sidebar.markdown("""
1. **Retrieve** — fetch top-4 docs from ChromaDB
2. **Grade** — filter noisy/irrelevant docs
3. **Rewrite** — if docs are bad, rewrite query and retry
4. **Generate** — Groq LLM generates answer from context
5. **Hallucination Check** — verify answer is grounded
6. **Regenerate** — if hallucination detected, try again
""")

st.sidebar.divider()
st.sidebar.markdown("### 💡 Sample Questions")
sample_questions = [
    "What is RAG and how does it work?",
    "Explain LangGraph and its key concepts",
    "What are vector embeddings?",
    "How does Groq differ from OpenAI?",
    "What is fine-tuning and when should I use it?",
    "Explain the Transformer architecture",
    "What is hallucination in LLMs?",
    "How does semantic re-ranking improve RAG?",
    "What is LLMOps?",
    "Explain agentic AI and the ReAct pattern",
]
for q in sample_questions:
    if st.sidebar.button(q, key=q):
        st.session_state["question_input"] = q

# ── Main Chat Area ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# Input
question = st.text_input(
    "Ask anything about AI, ML, or LLMs:",
    value=st.session_state.get("question_input", ""),
    placeholder="e.g. What is RAG and how does it address hallucination?",
    key="main_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("🚀 Ask", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=False):
        st.session_state.history = []
        st.rerun()

# ── Handle Question ────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    with st.spinner("🔄 Cyclic RAG pipeline running... (retrieve → grade → generate → verify)"):
        try:
            response = httpx.post(
                f"{API_BASE}/ask",
                json={"question": question},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.history.insert(0, {"question": question, "data": data})
            st.session_state.question_input = ""
        except Exception as e:
            st.error(f"Error: {e}")

# ── Display History ────────────────────────────────────────────────────────────
for item in st.session_state.history:
    q = item["question"]
    d = item["data"]

    with st.container():
        st.markdown(f"**❓ {q}**")

        # Pipeline transparency metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval Attempts", d["retrieval_attempts"])
        col2.metric("Docs Used", d["num_docs_used"])
        col3.metric(
            "Hallucination",
            "✅ None" if d["hallucination_flag"] == "no" else "⚠️ Detected"
        )
        col4.metric(
            "Query Rewritten",
            "Yes" if d["rewritten_question"] else "No"
        )

        if d.get("rewritten_question"):
            st.info(f"🔄 Query rewritten to: *{d['rewritten_question']}*")

        st.success(d["answer"])
        st.divider()
