"""
Chains — the LLM-powered components used by the graph nodes.

1. DocumentGrader   → is this document relevant to the question?
2. AnswerGenerator  → generate an answer from the documents
3. HallucinationGrader → does the answer stay grounded in the docs?
4. QueryRewriter    → rewrite the query for better retrieval
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


# ── 1. Document Relevance Grader ──────────────────────────────────────────────
GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document relevance grader.
Given a user question and a retrieved document, determine if the document contains information relevant to answering the question.

Respond with ONLY one word:
- 'yes' if the document is relevant
- 'no' if the document is not relevant

Do not explain. Do not add punctuation. Just one word."""),
    ("human", "Question: {question}\n\nDocument: {document}"),
])

def get_document_grader():
    return GRADER_PROMPT | get_llm() | StrOutputParser()


# ── 2. Answer Generator ───────────────────────────────────────────────────────
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant specializing in AI, ML, and LLM concepts.
Answer the user's question using ONLY the information provided in the context documents below.
Be thorough, accurate, and educational. If the context doesn't contain enough information, say so clearly.

Context:
{context}"""),
    ("human", "{question}"),
])

def get_generator():
    return GENERATOR_PROMPT | get_llm() | StrOutputParser()


# ── 3. Hallucination Grader ───────────────────────────────────────────────────
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hallucination detector.
Given a generated answer and the source documents it was based on, determine if the answer is grounded in the documents or if it contains hallucinated information not present in the sources.

Respond with ONLY one word:
- 'no' if the answer is grounded in the documents (no hallucination)
- 'yes' if the answer contains information not in the documents (hallucination detected)

Do not explain. Just one word."""),
    ("human", "Documents: {documents}\n\nGenerated Answer: {generation}"),
])

def get_hallucination_grader():
    return HALLUCINATION_PROMPT | get_llm() | StrOutputParser()


# ── 4. Query Rewriter ─────────────────────────────────────────────────────────
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query optimization expert.
The original query failed to retrieve relevant documents. Rewrite it to be more specific, 
use different terminology, and improve semantic clarity for vector search.
Return ONLY the rewritten query. No explanation, no preamble."""),
    ("human", "Original query: {question}"),
])

def get_query_rewriter():
    return REWRITER_PROMPT | get_llm() | StrOutputParser()
