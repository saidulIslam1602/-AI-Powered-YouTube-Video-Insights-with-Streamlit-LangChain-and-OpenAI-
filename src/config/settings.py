"""Application configuration settings."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings configuration."""
    
    # API Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo"
    max_tokens: int = 2000
    temperature: float = 0.1
    
    # Application Configuration
    app_name: str = "YouTube Video Insights"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 100
    similarity_k: int = 4
    max_video_length_minutes: int = 180  # 3 hours
    
    # Cache Configuration
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # UI Configuration
    page_title: str = "YouTube Video Insights"
    page_icon: str = "TV"
    layout: str = "centered"
    
    # Database Configuration (optional)
    database_url: str = ""
    api_key_required: bool = False
    api_key: str = ""
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, v):
        if not v:
            # Try to get from environment or use a default warning
            env_key = os.getenv("OPENAI_API_KEY", "")
            if env_key:
                return env_key
            # Don't raise error here, just warn in the app
            return ""
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings() 