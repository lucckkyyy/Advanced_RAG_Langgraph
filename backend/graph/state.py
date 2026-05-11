"""
Graph State — the shared memory that flows through every node in the RAG graph.
Every node reads from and writes to this state object.
"""
from typing import List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the RAG graph at any point in time.

    Attributes:
        question:        The original user question
        rewritten_question: Query after rewriting (if retrieval failed)
        documents:       List of retrieved document strings
        generation:      The LLM-generated answer
        retrieval_attempts: How many times we've tried to retrieve
        relevance_score: Whether retrieved docs are relevant
        hallucination_flag: Whether the answer has hallucinations
    """
    question: str
    rewritten_question: Optional[str]
    documents: List[str]
    generation: Optional[str]
    retrieval_attempts: int
    relevance_score: Optional[str]   # "yes" | "no"
    hallucination_flag: Optional[str]  # "yes" | "no"
