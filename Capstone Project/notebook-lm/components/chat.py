"""
Chat interface UI component.

Displays the conversation history and handles user input.
Each assistant message has a "Save as Note" button.
"""

import streamlit as st
from core.graph import run_graph
from utils.helpers import save_note_from_content


def render_chat():
    """Render the chat interface."""

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(
                            f"**{source['filename']}** — Page {source['page_number']}"
                        )
                        st.text(source["snippet"])

            # "Save as Note" button for assistant messages
            if message["role"] == "assistant":
                if st.button("Save as Note", key=f"save_note_{i}"):
                    note_file = save_note_from_content(message["content"])
                    st.toast(f"Saved: {note_file}")

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_graph(
                    query=prompt,
                    selected_docs=st.session_state.get("selected_docs", []),
                    web_search_enabled=st.session_state.get("web_search_enabled", False),
                )

                answer = result["answer"]
                sources = result.get("sources", [])

            st.markdown(answer)

            # Show sources
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.caption(
                            f"**{source['filename']}** — Page {source['page_number']}"
                        )
                        st.text(source["snippet"])

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
