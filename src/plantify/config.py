import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    ollama_host:str = Field("http://localhost:11434", min_length=1)
    text_model:str = Field("llama3.1:8b", min_length=1)
    vision_model:str = Field("qwen2.5vl:7b", min_length=1)

settings = Settings()