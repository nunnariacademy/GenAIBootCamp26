"""
Small utility functions used across the app.
"""

import os
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.prompts import SUMMARY_PROMPT
from config import LLM_MODEL, OLLAMA_BASE_URL, NOTES_DIR


def save_note_from_content(content: str) -> str:
    """
    Summarize content and save it as a markdown note.

    Args:
        content: The text to summarize and save.

    Returns:
        The filename of the saved note.
    """
    # Generate a summary
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"response": content})

    # Save to file
    timestamp = datetime.now()
    filename = f"note_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(NOTES_DIR, filename)

    note_content = f"""# Note — {timestamp.strftime('%B %d, %Y at %I:%M %p')}

{summary}

---
*Saved from chat*
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(note_content)

    return filename
