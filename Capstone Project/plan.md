# NotebookLM Replica — Capstone Project Plan

## Overview

Build a simplified NotebookLM clone using **Streamlit** + **Python**. The app lets users upload PDFs, chat with their documents using RAG, optionally search the web, and save notes from conversations. The project is split across 5 days, each introducing a new concept that layers onto the previous day's work.

**Audience:** Bootcamp students (final year). Keep everything simple, readable, and well-commented.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | Ollama — llama3.2:3b (local, free, no API key needed) |
| Framework | LangChain |
| Vector DB | ChromaDB (local, no setup needed) |
| Embeddings | Ollama — nomic-embed-text (local, pairs with Ollama setup) |
| Web Search | Tavily API (LangChain integration) |
| Orchestration | LangGraph |
| File Storage | Local filesystem |

---

## Project Structure

```
notebook-lm/
├── app.py                     # Main Streamlit app entry point
├── requirements.txt
├── .env                       # API keys (TAVILY_API_KEY) and Ollama config
├── config.py                  # Central config (model names, chunk sizes, paths)
│
├── components/                # Streamlit UI components
│   ├── sidebar.py             # PDF uploader, document list, chat settings
│   ├── chat.py                # Chat interface
│   └── notes.py               # Notes panel
│
├── core/                      # Backend logic
│   ├── prompts.py             # All system prompts and prompt templates
│   ├── document_processor.py  # PDF loading, chunking, embedding, metadata
│   ├── vector_store.py        # ChromaDB operations (store, query, filter)
│   ├── rag_chain.py           # RAG retrieval chain
│   ├── agents.py              # Agent with tools (search docs, web search, save notes)
│   └── graph.py               # LangGraph workflow orchestration
│
├── storage/                   # Local persistence
│   ├── uploads/               # Raw uploaded PDFs
│   ├── chroma_db/             # ChromaDB persistent storage
│   └── notes/                 # Saved markdown notes
│
└── utils/
    └── helpers.py             # Small utility functions
```

---

## App Layout

```
┌─────────────────────────────────────────────────────────────┐
│                        HEADER BAR                           │
├──────────────┬──────────────────────────┬───────────────────┤
│   SIDEBAR    │       CHAT AREA          │    NOTES TAB      │
│              │                          │                   │
│ [Upload PDF] │  User: "What does the    │  Saved Notes:     │
│              │   paper say about X?"    │                   │
│ Documents:   │                          │  ┌─────────────┐  │
│ ☑ paper1.pdf │  Bot: "According to      │  │ Note 1      │  │
│ ☑ paper2.pdf │   your document..."      │  │ Summary of  │  │
│ ☐ paper3.pdf │                          │  │ answer #3   │  │
│              │  [Save as Note] btn      │  └─────────────┘  │
│ ─────────── │                          │                   │
│ Chat Settings│                          │  [Download All]   │
│ ☐ Web Search │                          │                   │
├──────────────┴──────────────────────────┴───────────────────┤
│                        FOOTER                               │
└─────────────────────────────────────────────────────────────┘
```

- **Left sidebar:** PDF upload, document list with checkboxes (select which docs to query), chat settings toggle for web search.
- **Center:** Chat interface. Messages displayed in order. Each bot response has a "Save as Note" button.
- **Right:** Notes panel. Shows saved notes as collapsible cards. Option to download all as a single markdown file.

---

## Day-by-Day Build Plan

---

### DAY 1 — Prompt Engineering

**Concept taught:** System prompts, prompt templates, few-shot prompting, output formatting.

**What to build:**

1. **Streamlit skeleton** — Set up `app.py` with the 3-panel layout (sidebar, chat, notes placeholder).
2. **`config.py`** — Store model name (`llama3.2:3b`), Ollama base URL (`http://localhost:11434`), and any API keys loading from `.env`.
3. **`core/prompts.py`** — Write all the prompts the app will use:
   - `SYSTEM_PROMPT`: The main system prompt defining the assistant's role. It should behave like a notebook assistant — helpful, concise, always grounding answers in provided context.
   - `QA_PROMPT_TEMPLATE`: A prompt template with placeholders for `{context}` and `{question}`. This is the RAG prompt (context will be empty for now, filled in Day 3).
   - `SUMMARY_PROMPT`: For generating note summaries from a chat answer.
   - `WEB_SEARCH_PROMPT`: Instructions for when web search results are included.
4. **`components/chat.py`** — Build the chat interface using `st.chat_message` and `st.chat_input`. Store messages in `st.session_state.messages`.
5. **Direct LLM call** — Use Ollama's REST API directly via `requests` (no LangChain yet). Send user message + system prompt to `http://localhost:11434/api/chat`. Display the response. This teaches students that LLMs are just API endpoints before any framework magic.
6. **`components/sidebar.py`** — Placeholder for PDF uploader (UI only, no processing yet). Add a text area or controls to let students experiment with modifying the system prompt live and see how it changes responses.

**End of Day 1 result:** A working chat app powered by local Ollama with a well-crafted system prompt. Students can chat with the LLM and understand how prompt design affects output quality. The UI shell is in place.

**Key files created:** `app.py`, `config.py`, `core/prompts.py`, `components/chat.py`, `components/sidebar.py`, `requirements.txt`, `.env.example`

---

### DAY 2 — LangChain Ecosystem

**Concept taught:** LangChain document loaders, text splitters, chains, output parsers.

**What to build:**

1. **`core/document_processor.py`** — Build the PDF ingestion pipeline:
   - Use `PyPDFLoader` from LangChain to load uploaded PDFs.
   - Use `RecursiveCharacterTextSplitter` to chunk documents (chunk_size=1000, overlap=200).
   - Attach metadata to each chunk: `{filename, page_number, upload_date}`. This metadata is critical for Day 3 filtering.
   - Return processed chunks ready for embedding.
2. **`components/sidebar.py`** — Make the PDF uploader functional:
   - `st.file_uploader` accepting multiple PDFs.
   - On upload: save file to `storage/uploads/`, process with document_processor, show success message.
   - Display list of uploaded documents with checkboxes for selection.
3. **Migrate to LangChain** — Replace the raw Ollama REST call from Day 1:
   - Use `ChatOllama` (from `langchain_ollama`) as the LLM, pointing to `llama3.2:3b`.
   - Use `ChatPromptTemplate` with the prompts from Day 1.
   - Build a simple chain (prompt | llm | StrOutputParser).
4. **Show document info** — After upload, display chunk count and metadata in the sidebar so students can see what the splitter did.

**End of Day 2 result:** PDFs can be uploaded and processed into chunks with metadata. The chat now uses LangChain with `ChatOllama` instead of raw REST calls. No retrieval yet — the LLM still answers from its own knowledge.

**Key files created/modified:** `core/document_processor.py`, updated `components/sidebar.py`, updated `components/chat.py`

---

### DAY 3 — RAG System

**Concept taught:** Embeddings, vector stores, similarity search, retrieval chains, metadata filtering.

**What to build:**

1. **`core/vector_store.py`** — Set up ChromaDB:
   - Initialize a persistent ChromaDB client (stored in `storage/chroma_db/`).
   - Function to add document chunks with embeddings (use `OllamaEmbeddings` with `nomic-embed-text` model).
   - Function to query with metadata filtering: accept a list of selected filenames and filter results to only those documents using ChromaDB's `where` filter.
   - Function to list all stored documents (for the sidebar).
   - Function to delete a document's chunks by filename.
2. **Update `core/document_processor.py`** — After chunking, automatically embed and store chunks in ChromaDB.
3. **`core/rag_chain.py`** — Build the retrieval chain:
   - Take user query + list of selected document filenames.
   - Query ChromaDB with metadata filter (only selected docs).
   - Retrieve top-k relevant chunks (k=4).
   - Stuff retrieved chunks into the `{context}` placeholder of `QA_PROMPT_TEMPLATE`.
   - Pass to LLM and return the answer with source references.
   - Use LangChain's `create_retrieval_chain` or a simple `RunnableSequence`.
4. **Update `components/chat.py`** — Wire the chat to use the RAG chain instead of the plain LLM chain. Show source document references below each answer (which PDF, which page).
5. **Update `components/sidebar.py`** — Document checkboxes now actually control which documents are queried. Selecting/deselecting a document updates the metadata filter.

**End of Day 3 result:** The core NotebookLM experience works. Upload PDFs, select which ones to query, ask questions, get answers grounded in the documents with source citations.

**Key files created/modified:** `core/vector_store.py`, `core/rag_chain.py`, updated `core/document_processor.py`, updated `components/chat.py`, updated `components/sidebar.py`

---

### DAY 4 — AI Agents

**Concept taught:** Tool-calling agents, tool definitions, agent decision-making.

**What to build:**

1. **Web search tool** — Set up Tavily search:
   - Create a LangChain tool using `TavilySearchResults`.
   - The tool takes a query and returns web results.
   - This tool is only available when the user enables "Web Search" in chat settings.
2. **Document search tool** — Wrap the RAG chain from Day 3 as a tool:
   - The agent can call this tool to search uploaded documents.
   - Passes the selected document filter through.
3. **Note saving tool** — Create a tool that:
   - Takes a chat answer and generates a markdown summary using `SUMMARY_PROMPT`.
   - Saves the summary as a `.md` file in `storage/notes/`.
   - Returns confirmation to the user.
4. **`core/agents.py`** — Build the agent:
   - Use a ReAct agent pattern (`create_react_agent` from LangChain) — this works reliably with smaller local models like llama3.2:3b since it uses prompt-based reasoning rather than native tool-calling.
   - Register the tools: `[document_search, web_search, save_note]`.
   - The agent decides which tool to use based on the user's query and chat settings.
   - If web search is disabled in settings, don't include it in the tool list.
5. **`components/sidebar.py`** — Add the chat settings section:
   - A toggle/checkbox for "Enable Web Search".
   - Store in `st.session_state.web_search_enabled`.
6. **`components/notes.py`** — Build the notes panel:
   - Read saved notes from `storage/notes/`.
   - Display each note as a collapsible expander.
   - Add a "Download All Notes" button that combines all notes into one markdown file.
7. **Update `components/chat.py`** — Add a "Save as Note" button next to each assistant message. When clicked, triggers the note saving tool.

**End of Day 4 result:** The app now has an intelligent agent that chooses between searching documents, searching the web, or saving notes. The notes panel is functional.

**Key files created/modified:** `core/agents.py`, `components/notes.py`, updated `components/sidebar.py`, updated `components/chat.py`

---

### DAY 5 — LangGraph & MCP

**Concept taught:** Stateful graphs, conditional routing, node-based workflows, MCP tool integration.

**What to build:**

1. **`core/graph.py`** — Replace the Day 4 agent with a LangGraph workflow:
   - **Nodes:**
     - `classify_intent` — Determines if the query needs document search, web search, or both. Checks if web search is enabled.
     - `retrieve_documents` — Runs the RAG retrieval (from Day 3).
     - `web_search` — Runs Tavily search (from Day 4).
     - `generate_response` — Takes all retrieved context and generates the final answer.
     - `save_note` — Generates and saves a note when requested.
   - **Edges (conditional routing):**
     - After `classify_intent`: route to `retrieve_documents`, `web_search`, or both based on intent.
     - After retrieval nodes: always route to `generate_response`.
     - `save_note` is triggered separately by user action, not as part of the main query flow.
   - **State:** Define a `GraphState` TypedDict that carries: `query`, `selected_docs`, `web_search_enabled`, `retrieved_context`, `web_results`, `response`.
2. **Visualize the graph** — Use LangGraph's `.get_graph().draw_mermaid()` to display the workflow as a diagram in the app. This is a great teaching moment.
3. **MCP integration** — Demonstrate MCP by connecting one external tool:
   - Use a simple MCP server (e.g., filesystem MCP for reading/writing notes, or a custom one).
   - Show how MCP standardizes tool interfaces.
   - This can be a demonstration/discussion rather than a full integration if time is tight.
4. **Final polish:**
   - Error handling on all API calls.
   - Loading spinners during processing.
   - Clear chat / reset session button.
   - Clean up the UI spacing and labels.

**End of Day 5 result:** The full app is complete with LangGraph orchestrating the workflow. Students can see the graph visualization and understand how all pieces connect. MCP is introduced as the standard for tool communication.

**Key files created/modified:** `core/graph.py`, updated `app.py`, final polish on all components.

---

## Session State Schema

All app state lives in `st.session_state`:

```python
st.session_state = {
    "messages": [],                    # Chat history: [{"role": "user"/"assistant", "content": "..."}]
    "uploaded_docs": [],               # List of uploaded doc names: ["paper1.pdf", "paper2.pdf"]
    "selected_docs": [],               # Currently selected docs for querying
    "web_search_enabled": False,       # Chat setting toggle
    "notes": [],                       # Saved notes: [{"title": "...", "content": "...", "timestamp": "..."}]
    "processing": False,               # Loading state flag
}
```

---

## Key Design Decisions

1. **ChromaDB over FAISS** — Persistent, supports metadata filtering natively, zero config.
2. **Tavily over SerpAPI** — Built-in LangChain integration, simple API, good free tier.
3. **Local file storage for notes** — No database needed. Markdown files are human-readable and easy to debug.
4. **Streamlit session_state for chat** — No external database for chat history. Keeps it simple.
5. **Ollama + llama3.2:3b** — Completely free, runs locally, no API keys for LLM/embeddings. Students only need Ollama installed. Teaches them that AI apps don't require paid APIs.
6. **No authentication** — Single user app. No login, no multi-tenancy.

---

## Requirements

```
streamlit
langchain
langchain-ollama
langchain-community
langgraph
chromadb
pypdf
tavily-python
python-dotenv
requests
```

---

## Prerequisites

Students must have Ollama installed and running before Day 1:

```bash
# Install Ollama (https://ollama.com)
# Then pull the required models:
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify Ollama is running: `curl http://localhost:11434/api/tags` should return the model list.

---

## Environment Variables

```
TAVILY_API_KEY=tvly-...
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Build Order Summary

| Day | Concept | What Gets Added | What Students See |
|-----|---------|-----------------|-------------------|
| 1 | Prompt Engineering | Chat UI + system prompts + direct Ollama REST calls | A chatbot that responds based on prompt design |
| 2 | LangChain | PDF upload + processing + LangChain chains | Upload PDFs, see them chunked, chat via ChatOllama |
| 3 | RAG | Vector DB + retrieval + source citations | Ask questions, get answers FROM their documents |
| 4 | Agents | Web search + notes + tool-calling agent | Agent chooses tools, web search works, notes save |
| 5 | LangGraph & MCP | Graph orchestration + MCP + polish | Full app with visible workflow graph |
