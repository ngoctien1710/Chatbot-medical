from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data'
DEFAULT_PDF_SOURCE_ROOT = PROJECT_ROOT / 'Documents' / 'VietNam'
DEFAULT_CHAT_LOG_ROOT = PROJECT_ROOT / 'chat_log'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    service_host: str = Field(default='127.0.0.1', alias='RAG_SERVICE_HOST')
    service_port: int = Field(default=8001, alias='RAG_SERVICE_PORT')

    ollama_base_url: str = Field(default='http://127.0.0.1:11434', alias='OLLAMA_BASE_URL')
    chat_model: str = Field(default='mistral', alias='OLLAMA_CHAT_MODEL')
    embedding_model: str = Field(default='Dqdung205/medical_vietnamese_embedding', alias='OLLAMA_EMBED_MODEL')

    top_k: int = Field(default=5, alias='RAG_TOP_K')
    score_threshold: float = Field(default=0.35, alias='RAG_SCORE_THRESHOLD')
    chunk_size: int = Field(default=900, alias='RAG_CHUNK_SIZE')
    chunk_overlap: int = Field(default=180, alias='RAG_CHUNK_OVERLAP')
    chunk_token_size: int = Field(default=400, alias='RAG_CHUNK_TOKEN_SIZE')
    chunk_token_overlap: int = Field(default=80, alias='RAG_CHUNK_TOKEN_OVERLAP')
    min_chunk_chars: int = Field(default=200, alias='RAG_MIN_CHUNK_CHARS')

    data_root: Path = Field(default=DEFAULT_DATA_ROOT, alias='RAG_DATA_ROOT')
    data_raw_dir: Path = Field(default=DEFAULT_DATA_ROOT / 'raw', alias='RAG_DATA_RAW_DIR')
    vector_db_dir: Path = Field(default=DEFAULT_DATA_ROOT / 'chroma', alias='RAG_VECTOR_DB_DIR')
    pdf_source_dir: Path = Field(default=DEFAULT_PDF_SOURCE_ROOT, alias='RAG_PDF_SOURCE_DIR')

    request_timeout_seconds: int = Field(default=120, alias='RAG_REQUEST_TIMEOUT_SECONDS')
    chat_log_root: Path = Field(default=DEFAULT_CHAT_LOG_ROOT, alias='CHAT_LOG_ROOT')
    ocr_min_text_chars: int = Field(default=60, alias='OCR_MIN_TEXT_CHARS')
    ocr_dpi: int = Field(default=220, alias='OCR_DPI')
    clean_strip_toc: bool = Field(default=True, alias='CLEAN_STRIP_TOC')
    clean_strip_references: bool = Field(default=True, alias='CLEAN_STRIP_REFERENCES')
    glm_cleanup_enabled: bool = Field(default=True, alias='GLM_CLEANUP_ENABLED')
    glm_cleanup_model: str = Field(default='zai-org/GLM-5-FP8', alias='GLM_CLEANUP_MODEL')
    glm_cleanup_timeout_seconds: int = Field(default=45, alias='GLM_CLEANUP_TIMEOUT_SECONDS')
    glm_cleanup_max_chunk_chars: int = Field(default=2800, alias='GLM_CLEANUP_MAX_CHUNK_CHARS')
    glm_cleanup_max_chunks_per_doc: int = Field(default=12, alias='GLM_CLEANUP_MAX_CHUNKS_PER_DOC')
    hf_token: str = Field(default='', alias='HF_TOKEN')
    cors_origins: str = Field(
        default='http://127.0.0.1:5500,http://localhost:5500',
        alias='CORS_ORIGINS',
    )

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        if not origins:
            return ['*']
        if '*' in origins:
            return ['*']
        return origins


settings = Settings()
