import os

os.environ.setdefault("RAG_RETRIEVAL_URL", "http://retrieval.invalid")
os.environ.setdefault("RAG_INGESTION_URL", "http://ingestion.invalid")
os.environ.setdefault("AI_AGENT_URL", "http://agent.invalid")
os.environ.setdefault("AI_POSTGRES_DSN", "postgresql://test:test@postgres.invalid/test")
os.environ.setdefault("LAYA_URL", "http://laya.invalid")
