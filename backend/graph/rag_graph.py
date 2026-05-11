"""
RAG Graph — assembles all nodes and edges into the cyclic LangGraph.

Flow:
  retrieve → grade_documents →
    if relevant: generate → grade_hallucination →
      if grounded: END
      if hallucinating: generate (retry)
    if noisy + attempts left: rewrite → retrieve (loop back)
    if noisy + max attempts: generate (best effort)
"""
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    grade_hallucination_node,
)
from graph.edges import route_after_grading, route_after_hallucination_check


def build_rag_graph():
    """Build and compile the cyclic RAG graph."""

    workflow = StateGraph(GraphState)

    # ── Add Nodes ─────────────────────────────────────────────────────────────
    workflow.add_node("retrieve",            retrieve_node)
    workflow.add_node("grade_documents",     grade_documents_node)
    workflow.add_node("rewrite_query",       rewrite_query_node)
    workflow.add_node("generate",            generate_node)
    workflow.add_node("grade_hallucination", grade_hallucination_node)

    # ── Entry Point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    # ── Fixed Edges ───────────────────────────────────────────────────────────
    workflow.add_edge("retrieve",        "grade_documents")
    workflow.add_edge("rewrite_query",   "retrieve")          # ← THE CYCLE
    workflow.add_edge("generate",        "grade_hallucination")

    # ── Conditional Edges ─────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite":  "rewrite_query",
        },
    )

    workflow.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination_check,
        {
            "end":      END,
            "generate": "generate",
        },
    )

    return workflow.compile()


# Singleton — compiled once and reused across requests
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


def run_rag(question: str) -> dict:
    """
    Run the full cyclic RAG pipeline for a given question.

    Returns:
        dict with keys: answer, question, retrieval_attempts, hallucination_flag
    """
    graph = get_graph()

    initial_state: GraphState = {
        "question": question,
        "rewritten_question": None,
        "documents": [],
        "generation": None,
        "retrieval_attempts": 0,
        "relevance_score": None,
        "hallucination_flag": None,
    }

    print(f"\n{'='*60}")
    print(f"🚀 Starting Cyclic RAG for: '{question}'")
    print(f"{'='*60}")

    final_state = graph.invoke(initial_state)

    print(f"{'='*60}")
    print(f"✅ RAG complete — {final_state['retrieval_attempts']} retrieval attempt(s)")
    print(f"{'='*60}\n")

    return {
        "answer": final_state.get("generation", "No answer generated."),
        "question": question,
        "rewritten_question": final_state.get("rewritten_question"),
        "retrieval_attempts": final_state.get("retrieval_attempts", 1),
        "hallucination_flag": final_state.get("hallucination_flag", "no"),
        "num_docs_used": len(final_state.get("documents", [])),
    }
