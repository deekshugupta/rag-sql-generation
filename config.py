import os

# --- OpenAI ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))  # 1536 for -3-small, 3072 for -3-large
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

# --- Qdrant ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # None for local Qdrant
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "travel_rag_schema")

# --- Retrieval ---
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "8"))   # initial vector search width
TOP_N_FINAL = int(os.environ.get("TOP_N_FINAL", "3"))          # tables actually sent to the LLM
USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"
