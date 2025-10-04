"""
Shared application state for storing initialized async components
"""
from langgraph.graph.state import CompiledStateGraph

# Global state variable - will be initialized during app startup
dataframe_assistant_graph: CompiledStateGraph

async def initialize_assistant_graph():
    """Initialize the assistant graph during app startup"""
    global dataframe_assistant_graph
    
    from src.ai.autodf_ml_assistant import build_graph
    
    dataframe_assistant_graph = await build_graph()



def get_dataframe_assistant_graph() -> CompiledStateGraph:
    """Get the initialized dataframe assistant graph"""
    if dataframe_assistant_graph is None:
        raise RuntimeError("DataFrame Assistant graph not initialized. This should be called during app startup.")
    return dataframe_assistant_graph
