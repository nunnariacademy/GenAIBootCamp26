# Day-Wise Project Split — Student Tasks

Each day introduces one concept with one small, self-contained task. The tasks are intentionally scoped narrow so students don't feel overwhelmed. On Day 5, the full capstone code is handed out in class, so students don't need to assemble everything themselves.

---

## DAY 1 — Persona-Based Prompting with Prompt Templates

**Concept taught:** System prompts, prompt templates, persona-based / role-based prompting, how the same question produces different outputs under different roles.

**Student task:**

Build a small Python script (or Streamlit app) that lets the user pick a **persona** and then chat with an LLM under that persona.

1. Define at least **3 personas** using prompt templates, e.g.:
   - `Teacher` — explains concepts simply, uses analogies.
   - `Code Reviewer` — critical, points out issues, suggests improvements.
   - `Research Assistant` — concise, cites facts, neutral tone.
2. Use a **prompt template** with a `{persona_instructions}` placeholder and a `{user_question}` placeholder.
3. Add a **role switcher** (dropdown in Streamlit, or input in CLI) — changing the role re-renders the system prompt.
4. Send the composed prompt to Ollama (`llama3.2:3b`) via the REST API (`/api/chat`) using `requests`.
5. Ask the **same question** under different personas and observe how output changes.

**Deliverable:** One file (`day1_personas.py`) + a short note (3–5 lines) comparing outputs across personas.

**What they learn:** A prompt is just structured text. The "personality" of an LLM is 100% controlled by what you put in the system prompt.

---

## DAY 2 — Document Loading & Metadata Filtering

**Concept taught:** Document loaders, text splitters, metadata attachment, filtering chunks by metadata.

**Student task:**

Build a Python module that loads PDFs, splits them into chunks, attaches metadata, and lets you **filter chunks by metadata fields**.

1. Use `PyPDFLoader` from LangChain to load one or more PDFs.
2. Use `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) to split documents.
3. Attach metadata to each chunk:
   - `filename`
   - `page_number`
   - `upload_date`
   - `source_type` (e.g., "textbook", "paper", "notes" — student decides)
4. Write a small **filter function** `filter_chunks(chunks, **filters)` that returns only chunks matching the given metadata.
   - Example: `filter_chunks(chunks, filename="paper1.pdf", page_number=3)`
5. Test it by:
   - Loading 2 different PDFs.
   - Filtering to get only chunks from one file.
   - Filtering to get only chunks from a specific page range.

**Deliverable:** `day2_document_loader.py` with the loader + filter function, plus a small test block (`if __name__ == "__main__":`) showing 2–3 filter examples and their outputs.

**What they learn:** Chunks are not just text — they carry metadata, and metadata is what makes selective retrieval possible later.

---

## DAY 3 — Simple RAG with Semantic Search

**Concept taught:** Embeddings, vector stores, 

**Student task:**

Build a minimal RAG pipeline where retrieval is **simple**: semantic search (ChromaDB + embeddings)

1. Reuse Day 2's chunked documents.
2. **Semantic retriever:** Store chunks in ChromaDB using `OllamaEmbeddings` with `nomic-embed-text`. Build a retriever that returns top-k semantic matches.
5. Build a simple RAG chain:
   - Take a user question.
   - Retrieve top chunks using the  retriever.
   - Stuff them into a prompt template with `{context}` + `{question}`.
   - Send to `ChatOllama` and return the answer.
6. Compare side-by-side on the **same query**:
   - Semantic-only result
   - BM25-only result
   - Hybrid result

**Deliverable:** `day3_rag.py`- a rag based document retrieval and generation

**What they learn:** Semantic search
---

## DAY 4 — LLM-Based Tool Calling: Web Search + Summarization

**Concept taught:** Tool definitions, tool-calling, letting an LLM decide which tool to call, passing tool results back into the LLM.

**Student task:**

Write a tool-calling function where the LLM can choose between two tools: **web search** and **summarization**.

1. Define two tools (LangChain `@tool` decorator or plain function + schema):
   - `web_search(query: str)` — uses Tavily (`TavilySearchResults`) to fetch web results.
   - `summarize(text: str)` — sends the text to `ChatOllama` with a summary prompt and returns a short summary.
2. Use either:
   - A ReAct agent (`create_react_agent`) since llama3.2:3b doesn't reliably do native tool-calling, **or**
   - A manual loop: ask the LLM to respond in a JSON format like `{"tool": "web_search", "args": {"query": "..."}}`, parse it, call the tool, feed the result back.
3. Test with queries that require each tool:
   - `"What is the latest news on OpenAI?"` → should route to `web_search`.
   - `"Summarize this paragraph: ..."` → should route to `summarize`.
   - `"Find the latest news on X and summarize it"` → should call `web_search` then `summarize` (bonus).

**Deliverable:** `day4_tool_calling.py` with both tools and the agent/loop, plus example runs showing the tool the LLM picked for each query.

**What they learn:** Tools are just Python functions the LLM decides to invoke. Tool-calling is the foundation of agents.

---

## DAY 5 — Full Project Walkthrough (No Task)

**In class:** The complete capstone codebase (chat UI, PDF upload, hybrid RAG, agent, notes, LangGraph orchestration) is handed out and walked through live.

**Students:**
- Follow along as each module is explained.
- See how Days 1–4 concepts compose into the full app.
- Run the app locally, experiment with it.

**No homework / no task to submit.**

**What they learn:** How the small pieces they built on Days 1–4 fit into a real, production-shaped application.

---

## Summary Table

| Day | Concept | Scoped Task | File Deliverable |
|-----|---------|-------------|------------------|
| 1 | Persona-based prompting | Role switcher with prompt templates, direct Ollama calls | `day1_personas.py` |
| 2 | Document loading + metadata | PDF loader, chunker, metadata filter function | `day2_document_loader.py` |
| 3 | Hybrid RAG | Semantic + BM25 + Ensemble retriever + simple RAG chain | `day3_hybrid_rag.py` |
| 4 | Tool calling | Web search tool + summarize tool + LLM-driven tool selection | `day4_tool_calling.py` |
| 5 | Full project | Walkthrough of provided capstone codebase | — (no deliverable) |
