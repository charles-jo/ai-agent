from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant.qdrant.svc.cluster.local:6333"
    dense_collection: str = "iac_code_dense"
    sparse_collection: str = "iac_code_sparse"

    vllm_base_url: str = "http://vllm-router-service.vllm.svc.cluster.local/v1"
    vllm_api_key: str = "dummy"
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"

    search_top_k: int = 5
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    class Config:
        env_file = ".env"


settings = Settings()
