# 🧠 Production-Grade Cyclic RAG with LangGraph & Groq
## 📘 Building a Self-Correcting AI Retrieval System

I built this project to go beyond basic RAG and understand what makes a retrieval system truly production-grade. Standard RAG pipelines retrieve documents and generate answers in a straight line but what happens when the retrieved documents are noisy or irrelevant? What if the generated answer contains hallucinated information?

This project solves both problems by building a **cyclic, self-correcting RAG system** using LangGraph. The graph loops back on itself rewriting queries when retrieval fails, and regenerating answers when hallucinations are detected until it produces a grounded, accurate response.

---

## 🎯 Development Journey

### **Timeline & Approach**
- **Duration**: Built as part of a structured AI engineering portfolio series
- **Process**: Designed the graph state and nodes first, then wired conditional edges to create the correction loops
- **Focus**: LangGraph state machines, cyclic graph design, and LLM-based self-evaluation

---

## 🧠 Skills I Developed Through This Project

### **LangGraph & Graph-Based AI**
- Building stateful AI workflows using **LangGraph's StateGraph**
- Designing **cyclic graphs** where nodes loop back based on conditions
- Implementing **conditional edges** that route between nodes based on state
- Understanding the difference between linear chains and graph-based agentic systems
- Managing shared state with **TypedDict** across multiple graph nodes

### **Cyclic RAG Architecture**
- Designing a multi-stage RAG pipeline: retrieve → grade → rewrite → generate → verify
- Implementing **document relevance grading** to filter noisy documents
- Building **query rewriting** logic that improves retrieval on failed attempts
- Adding a **hallucination detection** layer that catches ungrounded answers
- Limiting retry loops with max attempt guards to prevent infinite cycles

### **Groq LLM Integration**
- Connecting **LangChain** to **Groq's LPU inference** for ultra-fast responses
- Using `langchain-groq` with `llama-3.1-8b-instant` for all grading and generation tasks
- Understanding how Groq's API is compatible with the OpenAI interface
- Designing prompts for binary classification tasks (yes/no graders)

### **ChromaDB & HuggingFace Embeddings**
- Building a persistent **ChromaDB** vector store with 12 AI/ML knowledge documents
- Using **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) for free local embeddings
- Seeding and querying the knowledge base with cosine similarity search
- Structuring documents with metadata for organized retrieval

### **FastAPI Backend**
- Building a clean REST API with a `/ask` endpoint that runs the full graph pipeline
- Returning structured responses including retrieval attempts, hallucination flags, and doc counts
- Implementing startup lifespan events for knowledge base seeding

### **Streamlit Frontend**
- Building a transparent AI interface showing pipeline metrics per query
- Displaying retrieval attempts, docs used, hallucination status, and query rewrites
- Adding sample questions and sidebar explanations of the cyclic flow

---

## ⚡ Technical Focus Areas

### **What I Built**
- A LangGraph cyclic graph with 5 nodes and conditional routing edges
- Document relevance grader that filters irrelevant docs before generation
- Query rewriter that improves search terms when retrieval fails
- Hallucination grader that verifies answers are grounded in source documents
- A ChromaDB knowledge base with 12 AI/ML concept documents
- FastAPI backend exposing the full graph as a REST endpoint
- Streamlit dashboard showing full pipeline transparency per query

### **Skills I Leveled Up**
- Graph-based agentic AI design with LangGraph
- Self-correcting AI pipeline architecture
- LLM-as-a-judge evaluation patterns
- Cyclic workflow design with loop guards
- Groq inference integration
- Production RAG beyond basic retrieve-and-generate

---

## 🏗️ System Architecture

```
User Query
    ↓
[Retrieve Node] ChromaDB semantic search
    ↓
[Grade Documents Node] LLM filters noisy docs
    ↓
Relevant? ──No──→ [Rewrite Query Node] ──→ [Retrieve Node] (cycle)
    │
   Yes
    ↓
[Generate Node] Groq LLM generates answer
    ↓
[Hallucination Grade Node] verifies answer is grounded
    ↓
Hallucinating? ──Yes──→ [Generate Node] (retry)
    │
   No
    ↓
Final Answer ✅
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Graph Framework | LangGraph |
| LLM | Groq- llama-3.1-8b-instant (free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free, local) |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions → GCP |

---

## 🗂️ Project Structure

```
advanced-rag-langgraph/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── graph/
│   │   ├── state.py             # LangGraph TypedDict state
│   │   ├── nodes.py             # 5 graph nodes
│   │   ├── edges.py             # Conditional routing logic
│   │   └── rag_graph.py        # Graph assembly + run_rag()
│   ├── chains/
│   │   └── chains.py            # Grader, generator, rewriter chains
│   ├── rag/
│   │   └── knowledge_base.py    # ChromaDB + HuggingFace embeddings
│   └── requirements.txt
├── frontend/
│   └── app.py                   # Streamlit dashboard
├── .github/workflows/
│   └── ci-cd.yml
├── docker-compose.yml
└── .env.example
```

---

## 🚀 The Learning Outcome

This project fundamentally changed how I think about RAG systems. Before this, I thought of RAG as a linear pipeline retrieve, generate, done. Building a cyclic graph made me understand that production AI systems need self-evaluation loops: the ability to recognize when retrieval has failed, rewrite the query, and try again. And the ability to detect hallucinations before serving an answer to a user.

LangGraph's conditional edges made this possible in a clean, modular way each node has a single responsibility, and the routing logic is separate from the business logic.

---

*Author: Aryan Rajguru | [Portfolio](https://aryanrajguru.com) | [LinkedIn](https://linkedin.com/in/aryanrajguru)*
