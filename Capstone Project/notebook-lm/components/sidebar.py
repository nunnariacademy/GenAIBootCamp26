"""
Sidebar UI component.

Contains:
- PDF file uploader
- Uploaded document list with selection checkboxes
- Chat settings (web search toggle)
- Document management (delete)
"""

import os
import streamlit as st
from core.document_processor import process_pdf
from core.vector_store import add_documents, list_stored_documents, delete_document
from config import UPLOADS_DIR


def render_sidebar():
    """Render the full sidebar with all its sections."""

    with st.sidebar:
        st.header("Documents")

        # --- PDF Upload Section ---
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Skip if already processed
                if uploaded_file.name in st.session_state.get("uploaded_docs", []):
                    continue

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save the file to disk
                    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Process: load → chunk → embed → store
                    chunks = process_pdf(file_path, uploaded_file.name)
                    num_chunks = add_documents(chunks)

                    # Track in session state
                    if "uploaded_docs" not in st.session_state:
                        st.session_state.uploaded_docs = []
                    st.session_state.uploaded_docs.append(uploaded_file.name)

                    st.success(f"{uploaded_file.name} — {num_chunks} chunks indexed")

        # --- Document List with Checkboxes ---
        st.divider()
        stored_docs = list_stored_documents()

        if stored_docs:
            st.subheader("Select Documents to Query")

            # Initialize selected_docs if needed
            if "selected_docs" not in st.session_state:
                st.session_state.selected_docs = stored_docs.copy()

            selected = []
            for doc_name in stored_docs:
                checked = st.checkbox(
                    doc_name,
                    value=doc_name in st.session_state.selected_docs,
                    key=f"doc_{doc_name}",
                )
                if checked:
                    selected.append(doc_name)

            st.session_state.selected_docs = selected

            # Delete document button
            st.divider()
            doc_to_delete = st.selectbox(
                "Remove a document",
                options=[""] + stored_docs,
                format_func=lambda x: "Select..." if x == "" else x,
            )
            if doc_to_delete and st.button("Delete", type="secondary"):
                with st.spinner(f"Deleting {doc_to_delete}..."):
                    delete_document(doc_to_delete)
                    # Remove the uploaded file too
                    file_path = os.path.join(UPLOADS_DIR, doc_to_delete)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    # Update session state
                    if doc_to_delete in st.session_state.get("uploaded_docs", []):
                        st.session_state.uploaded_docs.remove(doc_to_delete)
                    if doc_to_delete in st.session_state.get("selected_docs", []):
                        st.session_state.selected_docs.remove(doc_to_delete)
                    st.success(f"Deleted {doc_to_delete}")
                    st.rerun()
        else:
            st.info("Upload PDFs to get started.")

        # --- Chat Settings ---
        st.divider()
        st.subheader("Chat Settings")

        st.session_state.web_search_enabled = st.toggle(
            "Enable Web Search",
            value=st.session_state.get("web_search_enabled", False),
            help="Allow the assistant to search the web for additional information.",
        )

        # --- Clear Chat ---
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
