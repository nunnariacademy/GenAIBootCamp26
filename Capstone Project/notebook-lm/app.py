"""
NotebookLM Replica — Main Streamlit App

A simplified NotebookLM clone that lets users:
- Upload PDFs and chat with their documents using RAG
- Optionally search the web for supplementary information
- Save notes from conversations

Run with: streamlit run app.py
"""

import streamlit as st
from components.sidebar import render_sidebar
from components.chat import render_chat
from components.notes import render_notes
from core.graph import get_graph_mermaid

# --- Page Configuration ---
st.set_page_config(
    page_title="NotebookLM",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State ---
defaults = {
    "messages": [],
    "uploaded_docs": [],
    "selected_docs": [],
    "web_search_enabled": False,
    "notes": [],
    "processing": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Header ---
st.title("NotebookLM")
st.caption("Upload PDFs, ask questions, and save notes — powered by local AI.")

# --- Sidebar ---
render_sidebar()

# --- Main Content Area: Chat + Notes ---
chat_tab, notes_tab, graph_tab = st.tabs(["Chat", "Notes", "Workflow Graph"])

with chat_tab:
    render_chat()

with notes_tab:
    render_notes()

with graph_tab:
    st.header("LangGraph Workflow")
    st.caption("This diagram shows how your query flows through the system.")
    try:
        mermaid_code = get_graph_mermaid()
        st.code(mermaid_code, language="mermaid")
    except Exception as e:
        st.error(f"Could not render graph: {e}")
