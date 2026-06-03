from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    PROJECT_NAME: str = "Predictor Backend"
    DEBUG_MODE: bool = False
    GEMINI_API_KEY: str = Field(..., description="API Key for google genai")
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    MAX_RETRY_COUNT: int = 3
    USE_MOCK_TOOLS: bool = False
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
