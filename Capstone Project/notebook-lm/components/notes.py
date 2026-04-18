"""
Notes panel UI component.

Displays saved notes as collapsible cards.
Provides a "Download All Notes" button.
"""

import os
import streamlit as st
from config import NOTES_DIR


def load_notes() -> list:
    """
    Read all saved notes from the storage/notes/ directory.

    Returns:
        List of dicts with "filename", "title", and "content".
    """
    notes = []
    if not os.path.exists(NOTES_DIR):
        return notes

    for filename in sorted(os.listdir(NOTES_DIR), reverse=True):
        if filename.endswith(".md"):
            filepath = os.path.join(NOTES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from the first line (# Title)
            lines = content.strip().split("\n")
            title = lines[0].lstrip("# ").strip() if lines else filename

            notes.append({
                "filename": filename,
                "title": title,
                "content": content,
            })

    return notes


def combine_all_notes(notes: list) -> str:
    """Combine all notes into a single markdown string for download."""
    parts = []
    for note in notes:
        parts.append(note["content"])
    return "\n\n---\n\n".join(parts)


def render_notes():
    """Render the notes panel."""

    st.header("Saved Notes")

    notes = load_notes()

    if not notes:
        st.info("No notes yet. Save notes from chat responses using the 'Save as Note' button.")
        return

    # Download all notes button
    combined = combine_all_notes(notes)
    st.download_button(
        label="Download All Notes",
        data=combined,
        file_name="all_notes.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()

    # Display each note as a collapsible card
    for note in notes:
        with st.expander(note["title"]):
            st.markdown(note["content"])

            # Delete individual note
            if st.button("Delete Note", key=f"del_{note['filename']}"):
                filepath = os.path.join(NOTES_DIR, note["filename"])
                os.remove(filepath)
                st.toast(f"Deleted {note['filename']}")
                st.rerun()
