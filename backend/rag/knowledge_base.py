"""
Knowledge Base — ChromaDB vector store with HuggingFace embeddings.
Topic: AI, ML, and LLM concepts — makes the RAG system meta and impressive.
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "ai_ml_knowledge_base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Knowledge Base Documents ──────────────────────────────────────────────────
AI_ML_DOCS = [
    {
        "id": "doc-001",
        "content": """Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Model (LLM) responses by retrieving relevant information from external knowledge bases before generating an answer. RAG addresses the hallucination problem in LLMs by grounding responses in factual, retrieved content. The standard RAG pipeline consists of three stages: indexing (chunking and embedding documents into a vector store), retrieval (finding the most semantically similar documents to a query), and generation (using the retrieved context to produce an accurate answer). RAG is particularly useful for domain-specific applications where the LLM's training data may be outdated or insufficient.""",
        "metadata": {"topic": "RAG", "category": "architecture"},
    },
    {
        "id": "doc-002",
        "content": """LangGraph is a framework built on top of LangChain for creating stateful, multi-actor applications with LLMs using graph-based workflows. Unlike linear LangChain chains, LangGraph allows cyclic graphs where nodes can loop back based on conditions. Key concepts include: State (a shared TypedDict passed between nodes), Nodes (Python functions that read and modify state), Edges (connections between nodes, including conditional edges), and the StateGraph class which compiles the graph. LangGraph is ideal for agentic AI systems that require planning, self-correction, and iterative refinement — such as cyclic RAG systems that re-retrieve when documents are irrelevant.""",
        "metadata": {"topic": "LangGraph", "category": "framework"},
    },
    {
        "id": "doc-003",
        "content": """Vector embeddings are numerical representations of text (or other data) in a high-dimensional space where semantic similarity corresponds to geometric proximity. Embedding models like all-MiniLM-L6-v2 from HuggingFace convert text into dense vectors of fixed dimensions (e.g., 384 dimensions). Cosine similarity is commonly used to measure the distance between two embedding vectors, with values closer to 1 indicating higher similarity. Vector databases like ChromaDB, Pinecone, and Weaviate are optimized for storing and querying these high-dimensional vectors efficiently using Approximate Nearest Neighbor (ANN) algorithms like HNSW.""",
        "metadata": {"topic": "Embeddings", "category": "core-concept"},
    },
    {
        "id": "doc-004",
        "content": """Groq is an AI inference company that developed the Language Processing Unit (LPU), a specialized chip designed for ultra-fast LLM inference. Groq's API provides access to open-source models like Meta's LLaMA 3 (llama-3.1-8b-instant, llama-3.1-70b-versatile) and Mixtral with extremely low latency — typically under 500ms for short responses. The llama-3.1-8b-instant model is particularly popular for production applications requiring speed, offering a free tier with generous rate limits. Groq is compatible with the OpenAI SDK interface, making it easy to swap from OpenAI to Groq with minimal code changes.""",
        "metadata": {"topic": "Groq", "category": "infrastructure"},
    },
    {
        "id": "doc-005",
        "content": """Fine-tuning is the process of further training a pre-trained language model on a specific dataset to adapt it for a particular task or domain. Unlike RAG which retrieves external knowledge at inference time, fine-tuning bakes knowledge directly into the model weights. Techniques include: Full fine-tuning (updating all model parameters), LoRA (Low-Rank Adaptation — updating only small adapter matrices for efficiency), QLoRA (quantized LoRA for lower memory usage), and PEFT (Parameter-Efficient Fine-Tuning methods). Fine-tuning is best when you need the model to learn a specific style, format, or domain-specific reasoning pattern that can't be easily provided through prompting alone.""",
        "metadata": {"topic": "Fine-tuning", "category": "training"},
    },
    {
        "id": "doc-006",
        "content": """The Transformer architecture, introduced in the 2017 paper 'Attention Is All You Need', is the foundation of modern LLMs. Key components include: Self-Attention (allows each token to attend to all other tokens in the sequence), Multi-Head Attention (runs multiple attention operations in parallel to capture different relationships), Feed-Forward Networks (applied to each position independently), Positional Encoding (adds position information since transformers have no inherent sequence order), and Layer Normalization. The encoder processes input sequences while the decoder generates output sequences. Decoder-only transformers like GPT are used for text generation, while encoder-only models like BERT excel at understanding tasks.""",
        "metadata": {"topic": "Transformers", "category": "architecture"},
    },
    {
        "id": "doc-007",
        "content": """Agentic AI refers to AI systems where an LLM acts as a reasoning engine that autonomously plans, uses tools, and takes multi-step actions to achieve a goal. Key patterns include: ReAct (Reasoning + Acting — the agent alternates between thinking and tool use), Tool Calling (the LLM selects and invokes external tools like APIs, databases, or calculators), and Multi-Agent Systems (multiple specialized agents collaborate). Frameworks like LangChain, LangGraph, AutoGen, and CrewAI are popular for building agentic systems. The core challenge in agentic AI is reliability — ensuring the agent makes correct decisions, handles errors gracefully, and doesn't loop infinitely.""",
        "metadata": {"topic": "Agentic AI", "category": "architecture"},
    },
    {
        "id": "doc-008",
        "content": """Hallucination in LLMs refers to the phenomenon where the model generates confident-sounding but factually incorrect or fabricated information. Hallucinations occur because LLMs are trained to produce fluent, coherent text rather than factually accurate text. Mitigation strategies include: RAG (grounding responses in retrieved facts), Chain-of-Thought prompting (encouraging step-by-step reasoning), Hallucination graders (using an LLM to verify if the output is grounded in source documents), Constitutional AI (training models to self-critique), and Temperature reduction (lower temperature produces more deterministic outputs). Hallucination is one of the primary challenges preventing LLM deployment in high-stakes domains like medicine and law.""",
        "metadata": {"topic": "Hallucination", "category": "challenges"},
    },
    {
        "id": "doc-009",
        "content": """ChromaDB is an open-source, AI-native vector database designed for storing and querying embedding vectors. Key features include: Persistent storage (embeddings survive application restarts), Collections (named groups of embeddings with associated metadata), Built-in embedding functions (supports HuggingFace, OpenAI, and custom embedders), Metadata filtering (filter results by document attributes), and both local and client-server modes. ChromaDB supports cosine similarity and L2 distance for nearest-neighbor search. It is particularly popular for prototyping and production RAG applications due to its simplicity, zero infrastructure overhead in local mode, and Python-first API.""",
        "metadata": {"topic": "ChromaDB", "category": "infrastructure"},
    },
    {
        "id": "doc-010",
        "content": """Prompt engineering is the practice of designing and optimizing input prompts to elicit the best possible outputs from LLMs. Key techniques include: Zero-shot prompting (asking the model directly without examples), Few-shot prompting (providing 2-5 examples of the desired input-output format), Chain-of-Thought (CoT) prompting (asking the model to reason step-by-step), System prompts (setting the model's persona and constraints), and Structured output prompting (asking for JSON or specific formats). Advanced techniques include Self-Consistency (sampling multiple answers and taking the majority), Tree of Thoughts (exploring multiple reasoning paths), and ReAct (combining reasoning with actions). Prompt engineering significantly impacts LLM performance without requiring any model training.""",
        "metadata": {"topic": "Prompt Engineering", "category": "techniques"},
    },
    {
        "id": "doc-011",
        "content": """Semantic re-ranking is a technique used in advanced RAG systems to improve retrieval quality by re-scoring initially retrieved documents using a more powerful cross-encoder model. The standard RAG retrieval uses a bi-encoder (fast but less accurate) to retrieve top-K candidates. Re-ranking then uses a cross-encoder that jointly encodes the query and each document together, producing a more accurate relevance score. Models like Cohere Rerank, BGE-Reranker, and cross-encoder/ms-marco are popular choices. Re-ranking significantly improves answer quality by filtering out noisy documents before they reach the LLM, reducing hallucinations and improving response accuracy.""",
        "metadata": {"topic": "Semantic Re-ranking", "category": "techniques"},
    },
    {
        "id": "doc-012",
        "content": """LLMOps (Large Language Model Operations) is the set of practices for deploying, monitoring, and maintaining LLM-based applications in production. Key components include: Prompt versioning and management, LLM evaluation frameworks (DeepEval, RAGAS, TruLens) for measuring answer quality, Observability tools (Langfuse, LangSmith) for tracing LLM calls and debugging, Cost monitoring (tracking token usage and API costs), A/B testing for comparing different prompts or models, and Guardrails for safety and output validation. LLMOps extends traditional MLOps principles to address LLM-specific challenges like hallucination monitoring, prompt injection attacks, and the non-deterministic nature of LLM outputs.""",
        "metadata": {"topic": "LLMOps", "category": "operations"},
    },
]


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def seed_knowledge_base() -> None:
    collection = _get_collection()
    existing_ids = set(collection.get()["ids"])
    new_docs = [d for d in AI_ML_DOCS if d["id"] not in existing_ids]

    if not new_docs:
        print("✅ Knowledge base already seeded.")
        return

    collection.add(
        ids=[d["id"] for d in new_docs],
        documents=[d["content"] for d in new_docs],
        metadatas=[d["metadata"] for d in new_docs],
    )
    print(f"✅ Seeded {len(new_docs)} AI/ML documents into knowledge base.")


def retrieve_documents(query: str, n_results: int = 4) -> list[str]:
    """Retrieve top-N most relevant documents for a query."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results.get("documents", [[]])[0]
