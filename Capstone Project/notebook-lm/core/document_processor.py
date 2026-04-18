"""
PDF loading, chunking, and metadata attachment.

DAY 2 — LangChain Ecosystem:
This module handles the document ingestion pipeline:
1. Load PDF files using LangChain's PyPDFLoader
2. Split them into chunks using RecursiveCharacterTextSplitter
3. Attach metadata (filename, page number, upload date)
"""

from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> list:
    """
    Load a PDF file and return a list of LangChain Document objects.
    Each document represents one page of the PDF.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def chunk_documents(pages: list, filename: str) -> list:
    """
    Split pages into smaller chunks and attach metadata.

    Why chunk? LLMs have limited context windows, and smaller chunks
    give more precise retrieval results.

    Args:
        pages: List of Document objects from PyPDFLoader.
        filename: Original PDF filename (for metadata).

    Returns:
        List of Document objects, each representing a chunk.
    """
    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — keeping chunks semantically coherent.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Attach metadata to each chunk so we can filter and cite later
    upload_date = datetime.now().isoformat()
    for chunk in chunks:
        chunk.metadata["filename"] = filename
        chunk.metadata["upload_date"] = upload_date
        # page_number comes from PyPDFLoader automatically as "page"
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1  # 1-indexed

    return chunks


def process_pdf(file_path: str, filename: str) -> list:
    """
    Full pipeline: load PDF → chunk → attach metadata.

    Args:
        file_path: Path to the saved PDF file on disk.
        filename: Original filename (e.g., "paper1.pdf").

    Returns:
        List of chunked Document objects ready for embedding.
    """
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages, filename)
    return chunks
