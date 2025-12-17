"""
This script creates a Pydantic model to fix the personalization endpoint parameter issue.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ActivatePersonalizationRequest(BaseModel):
    chapter_id: str
    preferences: Optional[Dict[str, Any]] = None