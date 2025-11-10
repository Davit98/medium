import os
import stat
from typing import Any, List, Optional

import chromadb
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import nltk
from nltk.tokenize import word_tokenize

from src.logger import logger
from src.variables import TOP_K, VLLM_EMBEDDING_API_URL, vLLM_EMBEDDING_MODEL

nltk.download('punkt')
nltk.download('punkt_tab')

# Cache retrievers to avoid recreating them on every request
_chroma_retriever_cache = {}
_bm25_retriever_cache = {}


def get_chroma_retriever(chroma_persist_dir_path: str,
                         chroma_collection_name: str,
                         embedding_model: str = vLLM_EMBEDDING_MODEL,
                         documents: Optional[List[Document]] = None,
                         k: int = TOP_K,
                         **retriever_kwargs: Any) -> VectorStoreRetriever:
    """
    Initialize and return a Chroma-based vector store retriever using a specified embedding model.

    Parameters
    ----------
    chroma_persist_dir_path : str
        Path to the directory where Chroma should persist or load the vector store.

    chroma_collection_name : str
        Name of the Chroma collection to use or create.

    embedding_model : str, optional
        The name of the embedding model to for generating vector representations.
        Default is vLLM_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5".

    documents : List[Document], optional
        A list of LangChain `Document` objects to index if the vectorstore does not already exist.

    k : int, optional
        The number of top relevant documents to retrieve for a given query. Defaults to `TOP_K`.

    **retriever_kwargs : Any
        Additional keyword arguments to configure the retriever.

    Returns
    -------
    chroma_retriever : VectorStoreRetriever
        Instance that allows similarity-based querying over the Chroma vector store.
    """
    cache_key = (chroma_persist_dir_path, chroma_collection_name, embedding_model, k)
    if cache_key in _chroma_retriever_cache:
        logger.info("Using cached Chroma retriever")
        return _chroma_retriever_cache[cache_key]

    local_embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=VLLM_EMBEDDING_API_URL,
        api_key="dummy",
        check_embedding_ctx_length=False
    )
    if os.path.exists(chroma_persist_dir_path):
        logger.info(f"Loading collection=`{chroma_collection_name}` from ChromaDB")
        persistent_client = chromadb.PersistentClient(path=chroma_persist_dir_path)
        chroma_vectorstore = Chroma(
            client=persistent_client,
            collection_name=chroma_collection_name,
            embedding_function=local_embeddings,
        )
    else:
        logger.info(f"Creating a collection=`{chroma_collection_name}` in ChromaDB...")
        os.makedirs(chroma_persist_dir_path, exist_ok=True)
        os.chmod(chroma_persist_dir_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        
        chroma_vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=local_embeddings,
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir_path
        )
        logger.info(f"Process finished! Collection is persisted to `{chroma_persist_dir_path}`")

    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={'k': k}, **retriever_kwargs)
    _chroma_retriever_cache[cache_key] = chroma_retriever

    return chroma_retriever


def get_bm25_retriever(documents: List[Document], k: int = TOP_K) -> BM25Retriever:
    """
    Create and return a BM25Retriever initialized with the provided documents and top-k retrieval parameter.

    Parameters
    ----------
    documents : List[Document]
        Document objects to be indexed by the retriever.

    k : int, optional
        The number of top relevant documents to retrieve for a given query. Defaults to `TOP_K`.

    Returns
    -------
    bm25_retriever : BM25Retriever
        Instance configured with the provided documents and retrieval parameter.
    """
    if _bm25_retriever_cache:
        return _bm25_retriever_cache['bm25']

    bm25_retriever = BM25Retriever.from_documents(documents, k=k, preprocess_func=word_tokenize)
    _bm25_retriever_cache['bm25'] = bm25_retriever

    return bm25_retriever


async def hybrid_retrieval(query: str,
                     llm: ChatOpenAI,
                     chroma_persist_dir_path: str,
                     chroma_collection_name: str,
                     documents: Optional[List[Document]] = None,
                     previous_qa: Optional[str] = None,
                     embedding_model: str = vLLM_EMBEDDING_MODEL,
                     reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
                     alpha: float = 1.,
                     k: int = TOP_K,
                     **retriever_kwargs: Any) -> List[Document]:
    """
    Perform hybrid document retrieval by combining dense (using Chroma and embeddings) and sparse retrieval (BM25)
    methods with tunable weighting.

    Parameters
    ----------
    query : str
        The natural language query to retrieve the most relevant documents.

    llm : ChatOpenAI
        A vLLM backed language model used to optionally rewrite the query based on previous Q&A context.

    chroma_persist_dir_path : str
        Path to the directory where Chroma's vector store is persisted or should be created.

    chroma_collection_name : str
        Name of the Chroma collection for storing or loading dense embeddings.

    documents : List[Document], optional
        A list of documents to use for retrieval.

    previous_qa : str, optional
        The text of last couple of interactions, including both the human query and the AI's corresponding reply.

    embedding_model : str, optional
        The name of the embedding model to use for generating vector representations.
        Default is vLLM_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5".

    reranker_model : str, optional
        The name of a Hugging Face cross-encoder model, or any Hugging Face model that supports cross-encoder
        functionality, to be used as a reranker within the retriever. Default is "BAAI/bge-reranker-v2-m3".
        If not specified, reranking will not be applied.

    alpha : float, optional
        Weight for the dense retriever in the hybrid ensemble. The sparse retriever is weighted as (1 - alpha).
        Default is 1.0 (dense-only retrieval). The value must be between 0 and 1 inclusive.

    k : int, optional
        The number of top relevant documents to retrieve for a given query. Defaults to `TOP_K`. If reranker model is
        provided, k / 2 documents are returned.

    **retriever_kwargs : Any
        Additional keyword args to configure the dense retriever.

    Returns
    -------
    List[Document]
        A ranked list of documents retrieved using the hybrid retriever.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("`alpha` must be between 0 and 1 inclusive.")

    dense_retriever = get_chroma_retriever(
        chroma_persist_dir_path=chroma_persist_dir_path,
        chroma_collection_name=chroma_collection_name,
        embedding_model=embedding_model,
        documents=documents,
        k=k,
        **retriever_kwargs
    )

    if alpha != 1.:
        sparse_retriever = get_bm25_retriever(documents=documents, k=k)

        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[alpha, 1 - alpha]
        )
    else:
        retriever = dense_retriever

    if reranker_model:
        model = HuggingFaceCrossEncoder(model_name=reranker_model)
        compressor = CrossEncoderReranker(model=model, top_n=k//2)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    if previous_qa:
        updated_query = await llm.ainvoke(f"""
        Given a chat history and the latest human question which might reference context in the chat history, formulate 
        a standalone question which can be understood without the chat history. Do NOT answer the question, just 
        reformulate it if needed and otherwise return it as is. Return ONLY the reformulated question without any 
        explanation.
        
        <chat history>
        {previous_qa}
        <chat history>
        
        Human: {query}
        """)

        logger.debug(f"Transformed query for context-awareness: {updated_query.content}")

        docs = await retriever.ainvoke(updated_query.content)
    else:
        docs = await retriever.ainvoke(query)

    return docs
