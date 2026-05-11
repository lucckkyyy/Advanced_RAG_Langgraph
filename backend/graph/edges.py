"""
Edges — conditional routing logic for the LangGraph.
These functions decide which node to go to next based on the current state.
"""
from graph.state import GraphState

MAX_RETRIEVAL_ATTEMPTS = 2


def route_after_grading(state: GraphState) -> str:
    """
    After grading documents, decide:
    - If docs are relevant → generate answer
    - If docs are noisy AND we haven't hit max attempts → rewrite query
    - If docs are noisy AND max attempts reached → generate anyway (best effort)
    """
    relevance_score = state.get("relevance_score", "no")
    attempts = state.get("retrieval_attempts", 0)

    if relevance_score == "yes":
        print("➡️  Routing: relevant docs found → GENERATE")
        return "generate"
    elif attempts < MAX_RETRIEVAL_ATTEMPTS:
        print(f"➡️  Routing: noisy docs, attempt {attempts}/{MAX_RETRIEVAL_ATTEMPTS} → REWRITE")
        return "rewrite"
    else:
        print(f"➡️  Routing: max attempts reached → GENERATE (best effort)")
        return "generate"


def route_after_hallucination_check(state: GraphState) -> str:
    """
    After hallucination check, decide:
    - If no hallucination → return final answer
    - If hallucination detected → regenerate
    """
    hallucination_flag = state.get("hallucination_flag", "no")

    if hallucination_flag == "no":
        print("➡️  Routing: answer is grounded → END")
        return "end"
    else:
        print("➡️  Routing: hallucination detected → REGENERATE")
        return "generate"
