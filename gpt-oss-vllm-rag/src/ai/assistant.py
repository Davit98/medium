import os

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
import openai

from src.ai.rag.data import load_thesis_tex
from src.ai.rag.retriever import hybrid_retrieval
from src.logger import logger
from src.utils import format_docs
from src.variables import (
    ALPHA,
    REASONING_EFFORT,
    ROOT_DIR,
    SQLITE_CHECKPOINTER_DB_PATH,
    TEMPERATURE,
    TOP_K,
    VLLM_API_URL,
    vLLM_EMBEDDING_MODEL,
    vLLM_MODEL,
)

llm = ChatOpenAI(
    base_url=VLLM_API_URL,
    api_key="dummy",
    model=vLLM_MODEL,
    temperature=TEMPERATURE,
    reasoning_effort=REASONING_EFFORT
)


class State(MessagesState):
    summary: str
    context: str


async def build_graph() -> CompiledStateGraph:
    """
    Build and compile an asynchronous conversational retrieval-augmented generation (RAG) graph.

    Returns
    -------
    graph : CompiledStateGraph
        A compiled asynchronous state graph that orchestrates the RAG chatbot’s logic and 
        handles conversation summarization, retrieval, and synthesis.
    """
    summary_thr = 10
    n_msg_overlap = 2


    def synthesizer_system_message(context: str, summary_msg: str):
        prompt = f"""You are a helpful chatbot. 
        Use the information in the <context> section to answer the question. 

        <context>
        {context}
        <context>

        {summary_msg}
        """
        sys_msg = SystemMessage(content=prompt)
        return sys_msg

    # Node
    async def conversation_node(state: State):
        n_msg = len(state["messages"])

        summary = state.get("summary")
        if n_msg > summary_thr and (n_msg - 1) % summary_thr == 0:
            messages = state["messages"][-(summary_thr + 1):]

            conversation_history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_history.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage) and not msg.tool_calls:
                    conversation_history.append(f"AI: {msg.content}")

            human_ai_messages = "\n".join([item for item in conversation_history[:-1]])
            if summary:
                summary_message = f"""
                This is a summary of the conversation so far: '{summary}'

                Update and extend the summary by incorporating only the most important points and outcomes from the new
                messages below. Keep the overall summary brief and focused. Limit the total length to 3-4 sentences.
                <start>
                {human_ai_messages}
                <end>
                """
            else:
                summary_message = f"""
                Generate a brief summary (3-4 sentences) of the conversation below, capturing only the most important 
                points and outcomes. Exclude minor details and direct quotes.
                <start>
                {human_ai_messages}
                <end>
                """

            response = await llm.ainvoke(summary_message)

            return {"summary": response.content}

    # Node
    async def vector_db_node(state: State):
        messages = state["messages"]
        n_msg = len(messages)

        previous_qa = None
        if n_msg >= 3:
            q = max(-5, -n_msg)  # -5 means at most the last 2 interaction, -7 will mean at most 3, etc.

            previous_qa = f"""
            Previous Q&A:
            """
            for i in range(q, -1, 2):
                previous_qa = f"""{previous_qa}
                Human: {messages[i].content}
                AI: {messages[i + 1].content}
                """

        documents = None
        chroma_persist_dir_path = (ROOT_DIR / "phd-thesis").as_posix()
        if not os.path.exists(chroma_persist_dir_path) or ALPHA != 1.:
            documents = load_thesis_tex() 
  
        docs = await hybrid_retrieval(
            query=messages[-1].content,
            llm=llm,
            chroma_persist_dir_path=chroma_persist_dir_path,
            chroma_collection_name="phd-thesis",
            documents=documents,
            previous_qa=previous_qa,
            embedding_model=vLLM_EMBEDDING_MODEL,
            reranker_model=None,
            alpha=ALPHA,
            k=TOP_K
        )

        return {"context": format_docs(docs)}

    # Node
    async def synthesizer_node(state: State):
        n_msg = len(state["messages"])

        if n_msg < summary_thr:
            messages = state["messages"]
        else:
            idx = (n_msg % summary_thr) + n_msg_overlap
            messages = state["messages"][-idx:]

        if summary := state.get("summary"):
            summary_prompt = f"""
            For the reference here is a summary of conversation earlier: '{summary}'
            """
        else:
            summary_prompt = ""

        context = state.get("context")
        prompt = [synthesizer_system_message(context, summary_prompt)] + messages

        try:
            result = await llm.ainvoke(prompt)
        except openai.APIError as e:
            logger.error(f"OpenAI API error occured while invoking the LLM: {e}")
            result = "I apologize, but I couldn't generate a response to your question. Could you please rephrase or provide more context?"

        return {"messages": result}

    # Graph
    builder = StateGraph(State)

    # Define nodes: these do the work
    builder.add_node("conversation", conversation_node)
    builder.add_node("vector_db_node", vector_db_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "conversation")
    builder.add_edge("conversation", "vector_db_node")
    builder.add_edge("vector_db_node", "synthesizer")
    builder.add_edge("synthesizer", END)

    # Sqlite checkpointer
    sqlite_db_path = SQLITE_CHECKPOINTER_DB_PATH.as_posix()
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
    aconn = await aiosqlite.connect(sqlite_db_path)
    saver = AsyncSqliteSaver(aconn)
    
    graph = builder.compile(checkpointer=saver)

    return graph
