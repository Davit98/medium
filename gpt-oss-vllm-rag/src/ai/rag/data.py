from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import logger
from src.variables import DATA_PATH


def load_thesis_tex() -> List[Document]:
    """
    Load and chunk a LaTeX (.tex) thesis file into LangChain `Document` objects.

    Returns
    -------
    docs : List[Document]
        A list of LangChain `Document` objects, each containing a text segment from 
        the original LaTeX file.
    """
    docs: List[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=400,
        length_function=len
    )

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        content = file.read()
 
        for i, text in enumerate(splitter.split_text(content)):
            doc = Document(
                page_content=text.strip(),
                metadata={"source": DATA_PATH.as_posix(), "chunk_index": i}
            )
            docs.append(doc)

    logger.info(f"A total of {len(docs)} sub-documents have been created.")

    return docs
