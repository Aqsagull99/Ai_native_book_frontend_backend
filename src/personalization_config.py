"""
Configuration settings for the AI Personalization feature.
"""
import os
from typing import Optional

class PersonalizationConfig:
    """
    Configuration class for AI personalization service settings.
    """

    # OpenRouter API Configuration (reuse from translation)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free")

    # Personalization Timeout
    PERSONALIZATION_TIMEOUT: int = int(os.getenv("PERSONALIZATION_TIMEOUT", "30"))  # seconds

    # System Prompt for Personalization Agent
    PERSONALIZATION_SYSTEM_PROMPT: str = """
You are an AI assistant that personalizes educational content about robotics and AI.

Your task is to adapt content based on the user's reading level while STRICTLY preserving:
- All headings and subheadings (exact text, exact hierarchy)
- All code blocks (unchanged - do not modify anything inside ``` or ` backticks)
- All images and diagrams (keep all img tags and image references)
- The overall document structure and HTML formatting
- All factual information (no fabrication)

Reading Level Guidelines:
- BEGINNER: Simplify language, use more accessible vocabulary, add brief explanations for complex concepts
- INTERMEDIATE: Keep balanced explanations, moderate technical language
- ADVANCED: Use professional/concise language, minimal explanations, assume reader expertise

Technical Explanations Setting:
- If ENABLED: Add brief inline parenthetical explanations for technical terms
  Example: "ROS2 middleware" → "ROS2 (Robot Operating System 2, a framework for building robot applications) middleware"
- If DISABLED: Keep original terms without modification

Example Density Setting:
- MINIMAL: Remove supplementary examples, keep only essential code demonstrations
- NORMAL: Keep original examples unchanged
- DETAILED: Add additional clarifying examples where concepts are complex

CRITICAL RULES YOU MUST FOLLOW:
1. NEVER change heading text (h1, h2, h3, h4, h5, h6 tags)
2. NEVER modify content inside code blocks (```...``` or `...`)
3. NEVER add information not present or implied in original content
4. NEVER remove entire sections or paragraphs completely
5. NEVER change links, URLs, or references
6. Output must be valid HTML that matches the input structure
7. Preserve all HTML attributes (classes, ids, etc.)
8. Keep all special characters and formatting intact

Your output should ONLY be the personalized HTML content. Do not include any explanations, prefixes, or markdown formatting around it.
"""

    # Reading Level Specific Prompts
    BEGINNER_PROMPT_ADDITION: str = """
For BEGINNER level:
- Replace complex vocabulary with simpler alternatives
- Break down long sentences into shorter ones
- Add brief parenthetical clarifications for technical terms
- Ensure concepts are accessible to someone new to robotics
"""

    INTERMEDIATE_PROMPT_ADDITION: str = """
For INTERMEDIATE level:
- Maintain a balance between accessibility and technical depth
- Keep most technical terms but provide context where helpful
- Assume basic programming knowledge
"""

    ADVANCED_PROMPT_ADDITION: str = """
For ADVANCED level:
- Use precise technical terminology
- Be concise and direct
- Assume the reader has robotics and programming expertise
- Remove redundant explanations
"""

    @classmethod
    def get_full_prompt(cls, reading_level: str, technical_explanations: bool, example_density: str) -> str:
        """
        Build the complete system prompt based on personalization settings.

        Args:
            reading_level: 'beginner', 'intermediate', or 'advanced'
            technical_explanations: Whether to add inline explanations for technical terms
            example_density: 'minimal', 'normal', or 'detailed'

        Returns:
            Complete system prompt string
        """
        prompt = cls.PERSONALIZATION_SYSTEM_PROMPT

        # Add reading level specific instructions
        if reading_level == 'beginner':
            prompt += "\n" + cls.BEGINNER_PROMPT_ADDITION
        elif reading_level == 'intermediate':
            prompt += "\n" + cls.INTERMEDIATE_PROMPT_ADDITION
        elif reading_level == 'advanced':
            prompt += "\n" + cls.ADVANCED_PROMPT_ADDITION

        # Add technical explanations instruction
        if technical_explanations:
            prompt += "\n\nTechnical Explanations: ENABLED - Add parenthetical explanations for technical terms."
        else:
            prompt += "\n\nTechnical Explanations: DISABLED - Keep original terms without additional explanations."

        # Add example density instruction
        prompt += f"\n\nExample Density: {example_density.upper()}"
        if example_density == 'minimal':
            prompt += " - Show only essential examples, reduce supplementary demonstrations."
        elif example_density == 'normal':
            prompt += " - Keep all original examples as they are."
        elif example_density == 'detailed':
            prompt += " - Enhance examples with additional clarifications where helpful."

        return prompt

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate that required configuration values are present.
        """
        if not cls.OPENROUTER_API_KEY:
            # Don't raise error on import, just return False
            # This allows the app to start even if personalization is not configured
            return False
        return True

    @classmethod
    def is_configured(cls) -> bool:
        """
        Check if the personalization service is properly configured.
        """
        return bool(cls.OPENROUTER_API_KEY)
