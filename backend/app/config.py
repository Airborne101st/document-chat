from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    # Required settings
    gemini_api_key: str

    # Document processing settings
    max_page_limit: int = 50
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100

    # Retrieval settings
    top_k_chunks: int = 3

    # ChromaDB settings
    chroma_persist_dir: str = "./chroma_data"

    # File upload settings
    allowed_extensions: list[str] = ["pdf", "txt", "docx"]
    max_file_size_mb: int = 20

    # Application settings
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()


def get_settings() -> Settings:
    """
    FastAPI dependency function to get settings instance.

    Returns:
        The global Settings instance
    """
    return settings
