from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_PATH = ROOT_DIR / "data/mini-wikipedia.zip"

SQLITE_CHECKPOINTER_DB_PATH = ROOT_DIR / "state_db/conversation.db"

VLLM_API_URL = "http://localhost:3001/v1"
VLLM_EMBEDDING_API_URL = "http://localhost:7474/v1"

vLLM_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

TOP_K = 3  # top k most relevant items to retrieve
ALPHA = 1.  # weight for the dense retriever in the hybrid ensemble

vLLM_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.7
REASONING_EFFORT = "low" # choose between 'low', 'medium', and 'high'
