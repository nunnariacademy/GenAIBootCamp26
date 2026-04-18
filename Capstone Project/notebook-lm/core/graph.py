"""
LangGraph workflow orchestration.

DAY 5 — LangGraph:
Instead of a single agent making all decisions, LangGraph lets us define
a GRAPH of nodes (steps) with conditional edges (routing logic).
This gives us more control and visibility into the workflow.

Graph structure:
    classify_intent → retrieve_documents → generate_response
                    ↘ web_search ↗
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, WEB_SEARCH_PROMPT
from core.vector_store import query_documents
from core.rag_chain import format_context, format_sources
from config import LLM_MODEL, OLLAMA_BASE_URL, TAVILY_API_KEY


# ============================================================
# STATE DEFINITION
# This TypedDict defines what data flows through the graph.
# Each node can read from and write to this shared state.
# ============================================================

class GraphState(TypedDict):
    query: str
    selected_docs: list
    web_search_enabled: bool
    intent: Optional[str]           # "documents", "web", "both", or "general"
    retrieved_context: Optional[str]
    retrieved_sources: Optional[list]
    web_results: Optional[str]
    response: Optional[str]


# ============================================================
# NODE FUNCTIONS
# Each function is a node in the graph. It receives the current
# state and returns updates to the state.
# ============================================================

def classify_intent(state: GraphState) -> dict:
    """
    Determine what type of search the query needs.

    Key rule: if the user has documents selected, ALWAYS search them.
    The LLM only decides whether to ALSO search the web.
    This prevents the small model from accidentally skipping retrieval.
    """
    has_docs = bool(state.get("selected_docs"))
    web_enabled = state.get("web_search_enabled", False)

    if has_docs and web_enabled:
        # Let the LLM decide if web search would also help
        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        prompt = ChatPromptTemplate.from_template(
            """Would this question benefit from a web search in addition to \
searching the user's documents? Reply with ONLY "yes" or "no".

Question: {query}

Answer:"""
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"query": state["query"]}).strip().lower()
        intent = "both" if "yes" in answer else "documents"
    elif has_docs:
        # Documents selected, no web — always search documents
        intent = "documents"
    elif web_enabled:
        # No documents but web is on — search the web
        intent = "web"
    else:
        # No documents, no web — general chat
        intent = "general"

    return {"intent": intent}


def retrieve_documents(state: GraphState) -> dict:
    """
    Search the vector store for relevant document chunks.
    """
    results = query_documents(state["query"], state.get("selected_docs"))
    context = format_context(results)
    sources = format_sources(results)
    return {
        "retrieved_context": context,
        "retrieved_sources": sources,
    }


def search_web(state: GraphState) -> dict:
    """
    Search the web using Tavily.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    search = TavilySearchResults(
        max_results=3,
        tavily_api_key=TAVILY_API_KEY,
    )
    results = search.invoke(state["query"])

    # Format into readable text
    parts = []
    for r in results:
        parts.append(f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}")

    return {"web_results": "\n\n---\n\n".join(parts) if parts else "No web results found."}


def generate_response(state: GraphState) -> dict:
    """
    Generate the final response using all available context.
    """
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    context = state.get("retrieved_context", "")
    web_results = state.get("web_results", "")

    # Choose the appropriate prompt based on available context
    if context and web_results:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", WEB_SEARCH_PROMPT),
        ])
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "web_results": web_results,
            "question": state["query"],
        })
    elif context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", QA_PROMPT_TEMPLATE),
        ])
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": state["query"],
        })
    else:
        # General question — no context needed
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        response = (prompt | llm | StrOutputParser()).invoke({
            "question": state["query"],
        })

    return {"response": response}


# ============================================================
# ROUTING FUNCTIONS
# These decide which node to go to next based on the state.
# ============================================================

def route_by_intent(state: GraphState) -> str:
    """Route to the appropriate search node(s) based on classified intent."""
    intent = state.get("intent", "documents")
    if intent == "documents":
        return "retrieve_documents"
    elif intent == "web":
        return "search_web"
    elif intent == "both":
        return "retrieve_documents"  # Documents first, then web
    else:
        return "generate_response"   # General questions skip retrieval


def route_after_documents(state: GraphState) -> str:
    """After document retrieval, check if we also need web search."""
    if state.get("intent") == "both":
        return "search_web"
    return "generate_response"


# ============================================================
# GRAPH CONSTRUCTION
# Wire everything together into a LangGraph StateGraph.
# ============================================================

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Returns:
        A compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_response", generate_response)

    # Set the entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edges from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_documents": "retrieve_documents",
            "search_web": "search_web",
            "generate_response": "generate_response",
        },
    )

    # After document retrieval, maybe do web search too
    workflow.add_conditional_edges(
        "   ",
        route_after_documents,
        {
            "search_web": "search_web",
            "generate_response": "generate_response",
        },
    )

    # Web search always leads to generate_response
    workflow.add_edge("search_web", "generate_response")

    # Generate response is the final node
    workflow.add_edge("generate_response", END)

    return workflow.compile()


def run_graph(query: str, selected_docs: list = None, web_search_enabled: bool = False) -> dict:
    """
    Run the full LangGraph workflow.

    Args:
        query: The user's question.
        selected_docs: List of filenames to search.
        web_search_enabled: Whether web search is allowed.

    Returns:
        Dict with "response" and "sources".
    """
    graph = build_graph()

    initial_state = {
        "query": query,
        "selected_docs": selected_docs or [],
        "web_search_enabled": web_search_enabled,
        "intent": None,
        "retrieved_context": None,
        "retrieved_sources": None,
        "web_results": None,
        "response": None,
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result.get("response", "Sorry, I couldn't generate a response."),
        "sources": result.get("retrieved_sources", []),
    }


def get_graph_mermaid() -> str:
    """
    Get a Mermaid diagram of the workflow graph.
    Useful for visualizing the workflow in the UI.
    """
    graph = build_graph()
    return graph.get_graph().draw_mermaid()
