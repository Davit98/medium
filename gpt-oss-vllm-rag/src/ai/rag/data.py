import re
from typing import List
import zipfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import logger
from src.variables import DATA_PATH


def load_mini_wikipedia() -> List[Document]:
    """
    Load and process a mini Wikipedia text dataset into LangChain `Document` objects.

    Returns
    -------
    List[Document]
        A list of `Document` objects, each representing a text chunk with associated
        topic metadata.
    """
    docs: List[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200,
        length_function=len
    )
    with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if not file_name.startswith("__MACOSX") and file_name.endswith('txt'):
                with zip_ref.open(file_name) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    match = re.search(r'\b(\w+)\b', content)
                    topic = match.group(1) if match else "undefined"

                    for text in splitter.split_text(content):
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                'topic': topic
                            }
                        )
                        docs.append(doc)

    logger.info(f"A total of {len(docs)} sub-documents have been created.")

    return docs
