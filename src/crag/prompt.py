"""The fixed prompt template from the proposal.

The template is the same in every condition, so any change in the answer comes
from the passages. Closed-book (Condition 1) uses the same layout without the passages.
"""

RAG_TEMPLATE = """Use the passages below to answer the question.

{passages}

Question: {question}
Answer:"""

CLOSED_BOOK_TEMPLATE = """Answer the question.

Question: {question}
Answer:"""


def rag_prompt(question: str, passage_texts: list[str]) -> str:
    """Fill the RAG template (Conditions 2 and 3).

    Args:
        question: One wording of the fact's question.
        passage_texts: The 5 passage texts in slot order, some possibly edited.

    Returns:
        The prompt text, with the passages numbered "Passage 1:" to "Passage 5:".
    """
    numbered = [] # "Passage 1: ...", "Passage 2: ...", ...
    for i, text in enumerate(passage_texts, start=1): # Sweep over passages, counting from 1
        numbered.append(f"Passage {i}: {text}") # Label each passage with its slot number
    return RAG_TEMPLATE.format(passages="\n\n".join(numbered), question=question) # Fill the template, blank line between passages


def closed_book_prompt(question: str) -> str:
    """Fill the closed-book template (Condition 1).

    Args:
        question: One wording of the fact's question.

    Returns:
        The prompt text with no passages.
    """
    return CLOSED_BOOK_TEMPLATE.format(question=question) # Fill the template with the question only
