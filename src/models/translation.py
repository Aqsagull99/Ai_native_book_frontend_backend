"""
Translation request/response models for the Urdu Translation feature.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TranslateRequest(BaseModel):
    """
    Request model for translation API endpoint.
    """
    content: str  # The content to translate
    chapter_id: str  # The ID of the chapter being translated
    user_id: Optional[str] = None  # User ID (will be set by authentication middleware)
    preserve_formatting: bool = True  # Whether to preserve original formatting


class TranslateResponse(BaseModel):
    """
    Response model for translation API endpoint.
    """
    translated_content: str  # The translated content
    chapter_id: str  # The ID of the chapter that was translated
    user_id: str  # The ID of the user who requested the translation
    source_language: str = "English"  # The source language
    target_language: str = "Urdu"  # The target language
    translation_id: str  # Unique ID for this translation
    created_at: datetime  # When the translation was created


class TranslationStatusRequest(BaseModel):
    """
    Request model to check translation status.
    """
    translation_id: str


class TranslationStatusResponse(BaseModel):
    """
    Response model for translation status.
    """
    translation_id: str
    status: str  # "pending", "completed", "failed"
    translated_content: Optional[str] = None
    error_message: Optional[str] = None


class TranslationCacheRequest(BaseModel):
    """
    Request model for checking if translation is cached.
    """
    chapter_id: str
    user_id: str


class TranslationCacheResponse(BaseModel):
    """
    Response model for translation cache check.
    """
    is_cached: bool
    cached_content: Optional[str] = None
    chapter_id: str
    user_id: str