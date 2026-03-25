from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data'
DEFAULT_PDF_SOURCE_ROOT = PROJECT_ROOT / 'Documents' / 'VietNam'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    service_host: str = Field(default='127.0.0.1', alias='RAG_SERVICE_HOST')
    service_port: int = Field(default=8001, alias='RAG_SERVICE_PORT')

    ollama_base_url: str = Field(default='http://127.0.0.1:11434', alias='OLLAMA_BASE_URL')
    chat_model: str = Field(default='llama3.2:3b', alias='OLLAMA_CHAT_MODEL')
    embedding_model: str = Field(default='nomic-embed-text', alias='OLLAMA_EMBED_MODEL')

    top_k: int = Field(default=3, alias='RAG_TOP_K')
    score_threshold: float = Field(default=0.35, alias='RAG_SCORE_THRESHOLD')
    chunk_size: int = Field(default=900, alias='RAG_CHUNK_SIZE')
    chunk_overlap: int = Field(default=180, alias='RAG_CHUNK_OVERLAP')

    data_root: Path = Field(default=DEFAULT_DATA_ROOT, alias='RAG_DATA_ROOT')
    data_raw_dir: Path = Field(default=DEFAULT_DATA_ROOT / 'raw', alias='RAG_DATA_RAW_DIR')
    vector_db_dir: Path = Field(default=DEFAULT_DATA_ROOT / 'chroma', alias='RAG_VECTOR_DB_DIR')
    pdf_source_dir: Path = Field(default=DEFAULT_PDF_SOURCE_ROOT, alias='RAG_PDF_SOURCE_DIR')

    request_timeout_seconds: int = Field(default=120, alias='RAG_REQUEST_TIMEOUT_SECONDS')


settings = Settings()
