from typing import List

from langchain.schema import Document


def format_docs(docs: List[Document]):
    """
    Format a list of LangChain Document objects into a single string.

    Parameters
    ----------
    docs : List[Document]
        A list of LangChain Document objects to be formatted.

    Returns
    -------
    object : str
        A single string containing the concatenated `page_content` of all documents, separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)
