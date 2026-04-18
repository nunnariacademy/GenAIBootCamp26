"""
ChromaDB vector store operations.

DAY 3 — RAG System:
This module handles all interactions with ChromaDB:
- Storing document chunks with their embeddings
- Querying with semantic similarity search
- Filtering by document filename (metadata filtering)
- Listing and deleting documents
"""

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, TOP_K


def get_embedding_function():
    """
    Create an embedding function using Ollama's nomic-embed-text model.
    This converts text into numerical vectors that capture meaning.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store():
    """
    Get (or create) the persistent ChromaDB vector store.
    Data is saved to disk in storage/chroma_db/ so it survives app restarts.
    """
    return Chroma(
        collection_name="documents",
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DB_DIR,
    )


def add_documents(chunks: list) -> int:
    """
    Add document chunks to the vector store.

    Each chunk gets embedded (converted to a vector) and stored
    alongside its text and metadata.

    Args:
        chunks: List of LangChain Document objects with metadata.

    Returns:
        Number of chunks added.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def query_documents(query: str, selected_filenames: list = None, top_k: int = TOP_K) -> list:
    """
    Search the vector store for chunks relevant to the query.

    Uses MMR (Maximum Marginal Relevance) instead of plain similarity search.
    MMR balances relevance with diversity — so we get chunks from different
    parts of the document instead of multiple similar chunks (e.g., all from
    the references section).

    Args:
        query: The user's question.
        selected_filenames: List of filenames to filter by. If None, search all docs.
        top_k: Number of results to return.

    Returns:
        List of (Document, score) tuples, sorted by relevance.
    """
    vector_store = get_vector_store()

    # Build a metadata filter to only search selected documents
    filter_dict = None
    if selected_filenames and len(selected_filenames) > 0:
        if len(selected_filenames) == 1:
            filter_dict = {"filename": selected_filenames[0]}
        else:
            filter_dict = {"filename": {"$in": selected_filenames}}

    # MMR fetches more candidates (fetch_k) then picks top_k diverse results
    kwargs = {"k": top_k, "fetch_k": top_k * 4}
    if filter_dict:
        kwargs["filter"] = filter_dict

    docs = vector_store.max_marginal_relevance_search(query, **kwargs)

    # Wrap in (doc, score) tuples to keep the same interface
    # MMR doesn't return scores, so we use 0.0 as placeholder
    return [(doc, 0.0) for doc in docs]


def list_stored_documents() -> list:
    """
    Get a list of all unique document filenames stored in ChromaDB.

    Returns:
        List of filename strings.
    """
    vector_store = get_vector_store()
    collection = vector_store._collection

    # Get all metadata from the collection
    result = collection.get(include=["metadatas"])
    if not result["metadatas"]:
        return []

    # Extract unique filenames
    filenames = set()
    for metadata in result["metadatas"]:
        if "filename" in metadata:
            filenames.add(metadata["filename"])

    return sorted(list(filenames))


def delete_document(filename: str) -> int:
    """
    Delete all chunks belonging to a specific document.

    Args:
        filename: The filename to delete (e.g., "paper1.pdf").

    Returns:
        Number of chunks deleted.
    """
    vector_store = get_vector_store()
    collection = vector_store._collection

    # Find all chunk IDs for this filename
    result = collection.get(
        where={"filename": filename},
        include=["metadatas"],
    )

    if result["ids"]:
        collection.delete(ids=result["ids"])
        return len(result["ids"])

    return 0
