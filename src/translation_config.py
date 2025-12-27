"""
Configuration settings for the Urdu Translation feature.
"""
import os
from typing import Optional

class TranslationConfig:
    """
    Configuration class for translation service settings.
    """

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free")

    # Translation Settings
    TARGET_LANGUAGE: str = "Urdu"
    SOURCE_LANGUAGE: str = "English"

    # Caching Configuration
    CACHE_TTL_SECONDS: int = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))  # 1 hour default
    MAX_CACHE_SIZE: int = int(os.getenv("TRANSLATION_MAX_CACHE_SIZE", "1000"))

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("TRANSLATION_RATE_LIMIT_REQUESTS", "10"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("TRANSLATION_RATE_LIMIT_WINDOW", "60"))  # seconds

    # Translation Timeout
    TRANSLATION_TIMEOUT: int = int(os.getenv("TRANSLATION_TIMEOUT", "30"))  # seconds

    # System Prompt for Translation Agent
    TRANSLATION_SYSTEM_PROMPT: str = """
    You are an expert translator specializing in converting content to Urdu while preserving the original structure and formatting.
    Your task is to translate the provided content to Urdu while maintaining:

    1. All structural elements (headings, lists, code blocks, tables, quotes)
    2. Formatting and hierarchy
    3. Special technical terminology (keep technical terms in English if no appropriate Urdu equivalent exists)
    4. Code blocks and inline code must remain completely unchanged - DO NOT translate any content inside backticks or code blocks
    5. Mathematical expressions and formulas should remain unchanged
    6. Proper names, URLs, and references should remain unchanged
    7. Preserve the exact formatting of markdown elements like **bold**, *italic*, [links](url), etc.

    When you encounter code blocks marked with triple backticks (```), or inline code marked with single backticks (`),
    preserve them exactly as they are without any translation. The content inside these markers must remain in English.
    Always maintain the original document structure and ensure the translated content is readable and comprehensible in Urdu.
    """

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate that required configuration values are present.
        """
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for translation service")

        if not cls.OPENROUTER_MODEL:
            raise ValueError("OPENROUTER_MODEL is required for translation service")

        return True

# Validate configuration on import
TranslationConfig.validate_config()