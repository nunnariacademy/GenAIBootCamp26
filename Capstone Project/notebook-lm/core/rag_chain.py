"""
RAG (Retrieval-Augmented Generation) chain.

DAY 3 — RAG System:
This module ties together retrieval and generation:
1. Take the user's question + selected documents
2. Retrieve relevant chunks from ChromaDB
3. Stuff those chunks into the prompt template
4. Send to the LLM and return the answer with sources
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE
from core.vector_store import query_documents
from config import LLM_MODEL, OLLAMA_BASE_URL


def get_llm():
    """Create the ChatOllama LLM instance."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def format_context(results: list) -> str:
    """
    Format retrieved chunks into a single context string.

    Each chunk is labeled with its source document and page number
    so the LLM can cite them in its answer.
    """
    if not results:
        return "No relevant documents found."

    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        context_parts.append(
            f"[Source {i}: {filename}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def format_sources(results: list) -> list:
    """
    Extract source information from results for display in the UI.

    Returns:
        List of dicts with filename, page_number, and a snippet.
    """
    sources = []
    seen = set()
    for doc, score in results:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        key = f"{filename}-p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": filename,
                "page_number": page,
                "snippet": doc.page_content[:150] + "...",
            })
    return sources


def rag_query(question: str, selected_filenames: list = None) -> dict:
    """
    Run the full RAG pipeline: retrieve → format → generate.

    Args:
        question: The user's question.
        selected_filenames: Which documents to search.

    Returns:
        Dict with "answer" (str) and "sources" (list of source dicts).
    """
    # Step 1: Retrieve relevant chunks
    results = query_documents(question, selected_filenames)

    # Step 2: Format context
    context = format_context(results)
    sources = format_sources(results)

    # Step 3: Build the prompt and call the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", QA_PROMPT_TEMPLATE),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "sources": sources,
    }
