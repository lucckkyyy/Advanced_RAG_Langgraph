"""
Graph Nodes — each function is a node in the LangGraph.
Nodes read from GraphState, do work, and return updated state fields.
"""
from graph.state import GraphState
from rag.knowledge_base import retrieve_documents
from chains.chains import (
    get_document_grader,
    get_generator,
    get_hallucination_grader,
    get_query_rewriter,
)

MAX_RETRIEVAL_ATTEMPTS = 2


def retrieve_node(state: GraphState) -> dict:
    """
    Node 1: Retrieve documents from ChromaDB.
    Uses rewritten question if available, otherwise original question.
    """
    question = state.get("rewritten_question") or state["question"]
    attempts = state.get("retrieval_attempts", 0) + 1

    print(f"📚 [Retrieve Node] Attempt {attempts} — query: '{question[:60]}...'")

    documents = retrieve_documents(question)

    return {
        "documents": documents,
        "retrieval_attempts": attempts,
    }


def grade_documents_node(state: GraphState) -> dict:
    """
    Node 2: Grade each retrieved document for relevance.
    Filters out noisy/irrelevant documents.
    Sets relevance_score to 'yes' if any docs pass, 'no' if all fail.
    """
    question = state.get("rewritten_question") or state["question"]
    documents = state["documents"]

    print(f"🔍 [Grade Node] Grading {len(documents)} documents...")

    grader = get_document_grader()
    relevant_docs = []

    for doc in documents:
        score = grader.invoke({"question": question, "document": doc})
        score = score.strip().lower()
        if score == "yes":
            relevant_docs.append(doc)
            print(f"  ✅ Document relevant")
        else:
            print(f"  ❌ Document noisy/irrelevant — filtered out")

    relevance_score = "yes" if relevant_docs else "no"
    print(f"📊 Relevance result: {relevance_score} ({len(relevant_docs)}/{len(documents)} docs passed)")

    return {
        "documents": relevant_docs,
        "relevance_score": relevance_score,
    }


def rewrite_query_node(state: GraphState) -> dict:
    """
    Node 3: Rewrite the query for better retrieval.
    Called when retrieved documents were not relevant.
    """
    question = state["question"]
    print(f"✏️  [Rewrite Node] Rewriting query...")

    rewriter = get_query_rewriter()
    rewritten = rewriter.invoke({"question": question})
    rewritten = rewritten.strip()

    print(f"  Original : {question}")
    print(f"  Rewritten: {rewritten}")

    return {"rewritten_question": rewritten}


def generate_node(state: GraphState) -> dict:
    """
    Node 4: Generate an answer using Groq LLM + retrieved context.
    """
    question = state.get("rewritten_question") or state["question"]
    documents = state["documents"]

    print(f"🤖 [Generate Node] Generating answer with Groq...")

    context = "\n\n---\n\n".join(documents)
    generator = get_generator()
    generation = generator.invoke({"question": question, "context": context})

    print(f"  Generated {len(generation)} characters")

    return {"generation": generation}


def grade_hallucination_node(state: GraphState) -> dict:
    """
    Node 5: Check if the generated answer is grounded in the documents.
    Sets hallucination_flag to 'yes' if hallucination detected.
    """
    documents = state["documents"]
    generation = state["generation"]

    print(f"🧪 [Hallucination Node] Checking answer groundedness...")

    grader = get_hallucination_grader()
    docs_text = "\n\n".join(documents)
    flag = grader.invoke({"documents": docs_text, "generation": generation})
    flag = flag.strip().lower()

    # Normalize to yes/no
    if "yes" in flag:
        flag = "yes"
    else:
        flag = "no"

    print(f"  Hallucination detected: {flag}")

    return {"hallucination_flag": flag}
