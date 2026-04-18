"""
AI Agent with tools for document search, web search, and note saving.

DAY 4 — AI Agents:
An agent is an LLM that can decide WHICH tool to use based on the user's query.
Instead of hardcoding "always search documents", the agent reasons about
what action to take. This module defines:
- Three tools the agent can use
- The ReAct agent that orchestrates tool usage
"""

import os
from datetime import datetime
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from core.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
from core.rag_chain import rag_query
from config import LLM_MODEL, OLLAMA_BASE_URL, TAVILY_API_KEY, NOTES_DIR
# ============================================================
# TOOL DEFINITIONS
# Each tool is a function decorated with @tool.
# The docstring becomes the tool's description — the agent
# reads this to decide when to use each tool.
# ============================================================

@tool
def document_search(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query.
    Use this tool when the user asks about content in their uploaded PDFs.
    Returns relevant passages with source citations."""

    import streamlit as st
    selected = st.session_state.get("selected_docs", [])
    result = rag_query(query, selected)

    # Format sources for the agent
    source_text = ""
    if result["sources"]:
        source_text = "\n\nSources:\n"
        for s in result["sources"]:
            source_text += f"- {s['filename']}, Page {s['page_number']}\n"

    return result["answer"] + source_text


@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.
    Use this tool when the user's question requires up-to-date information
    that might not be in their documents, or when they explicitly ask
    to search the web."""

    from langchain_community.tools.tavily_search import TavilySearchResults

    search = TavilySearchResults(
        max_results=3,
        tavily_api_key=TAVILY_API_KEY,
    )
    results = search.invoke(query)

    # Format results into readable text
    output_parts = []
    for r in results:
        output_parts.append(f"**{r.get('url', 'N/A')}**\n{r.get('content', '')}")

    return "\n\n---\n\n".join(output_parts) if output_parts else "No web results found."


@tool
def save_note(content: str) -> str:
    """Save a piece of information as a note for later reference.
    Use this tool when the user asks to save, remember, or bookmark information.
    The content will be summarized and saved as a markdown note."""

    # Generate a summary using the LLM
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"response": content})

    # Save to a markdown file
    timestamp = datetime.now()
    filename = f"note_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(NOTES_DIR, filename)

    note_content = f"""# Note — {timestamp.strftime('%B %d, %Y at %I:%M %p')}

{summary}

---
*Saved automatically from chat*
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(note_content)

    return f"Note saved as {filename}"


# ============================================================
# AGENT SETUP
# ============================================================

# ReAct prompt template — this is the "brain" of the agent.
# It tells the LLM how to reason step-by-step and use tools.
REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


def get_agent(web_search_enabled: bool = False) -> AgentExecutor:
    """
    Create a ReAct agent with the appropriate tools.

    Args:
        web_search_enabled: Whether to include the web search tool.

    Returns:
        An AgentExecutor ready to handle queries.
    """
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    # Build tool list based on settings
    tools = [document_search, save_note]
    if web_search_enabled and TAVILY_API_KEY:
        tools.append(web_search)

    # Create the ReAct agent
    agent = create_react_agent(llm, tools, REACT_PROMPT)

    # AgentExecutor runs the agent loop: think → act → observe → repeat
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,         # Print reasoning steps (great for learning!)
        handle_parsing_errors=True,
        max_iterations=5,     # Safety limit to prevent infinite loops
    )

    return executor


def run_agent(query: str, web_search_enabled: bool = False) -> str:
    """
    Run the agent on a user query and return the final answer.

    Args:
        query: The user's question.
        web_search_enabled: Whether web search is available.

    Returns:
        The agent's final answer as a string.
    """
    executor = get_agent(web_search_enabled)
    result = executor.invoke({"input": query})
    return result["output"]
