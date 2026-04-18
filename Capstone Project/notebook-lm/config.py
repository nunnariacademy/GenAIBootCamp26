"""
Central configuration for the NotebookLM app.
All model names, paths, and tunable parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Tavily (Web Search) ---
TAVILY_API_KEY = "Use your key"

# --- Document Processing ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RAG Retrieval ---
TOP_K = 8  # Number of chunks to retrieve per query

# --- Storage Paths ---
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(STORAGE_DIR, "chroma_db")
NOTES_DIR = os.path.join(STORAGE_DIR, "notes")

# Make sure storage directories exist
for directory in [UPLOADS_DIR, CHROMA_DB_DIR, NOTES_DIR]:
    os.makedirs(directory, exist_ok=True)
