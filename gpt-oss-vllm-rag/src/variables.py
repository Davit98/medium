from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_PATH = ROOT_DIR / "data/main.tex"

SQLITE_CHECKPOINTER_DB_PATH = ROOT_DIR / "state_db/conversation.db"

VLLM_API_URL = "http://localhost:3001/v1"
VLLM_EMBEDDING_API_URL = "http://localhost:3002/v1"

vLLM_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

TOP_K = 2  # top k most relevant items to retrieve
ALPHA = 0.5  # weight for the dense retriever in the hybrid ensemble

vLLM_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.7
REASONING_EFFORT = "low" # choose between 'low', 'medium', and 'high'
