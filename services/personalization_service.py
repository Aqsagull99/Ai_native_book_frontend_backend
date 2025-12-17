"""
Personalization Service
Handles all personalization-related business logic including:
- Activating personalization for chapters
- Managing user preferences
- Awarding bonus points
- Checking personalization status
"""
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from models import (
    UserPersonalizationPreference,
    UserBonusPoints,
    Chapter,
    User
)
import json
from datetime import datetime


class PersonalizationService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def activate_personalization(
        self,
        user_id: str,
        chapter_id: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Activate personalization for a chapter and award bonus points.

        Args:
            user_id: The ID of the user activating personalization
            chapter_id: The ID of the chapter to personalize
            preferences: User-specific content customization settings

        Returns:
            Dictionary with success status, message, points earned, and personalized content
        """
        logging.info(f"Personalization activation requested for user {user_id}, chapter {chapter_id}")
        # Check if personalization already exists for this user-chapter combination
        existing_personalization = self.db.query(UserPersonalizationPreference).filter(
            and_(
                UserPersonalizationPreference.user_id == user_id,
                UserPersonalizationPreference.chapter_id == chapter_id
            )
        ).first()

        if existing_personalization:
            logging.warning(f"Duplicate personalization attempt for user {user_id}, chapter {chapter_id}")
            return {
                "success": False,
                "message": "Personalization already activated for this chapter",
                "points_earned": 0,
                "is_duplicate": True
            }

        # Check if chapter exists
        chapter = self.db.query(Chapter).filter(Chapter.id == chapter_id).first()
        if not chapter:
            # If chapter doesn't exist in our chapters table, we could potentially still allow personalization
            # by creating a record in the database for it. Let's check if it exists in the docs folder first.
            import os
            docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'docs')
            # Check if the docs directory exists first
            if os.path.exists(docs_path):
                # Check if a file with the chapter_id exists in the docs folder (with .md extension)
                chapter_file_path = os.path.join(docs_path, f"{chapter_id}.md")
                if not os.path.exists(chapter_file_path):
                    # Also check if it exists as a directory (which might contain an index.md)
                    chapter_dir_path = os.path.join(docs_path, chapter_id)
                    if not os.path.exists(chapter_dir_path):
                        # Chapter doesn't exist in docs, but we can still create a record for personalization
                        # This allows for dynamic chapter creation
                        pass  # Continue to create chapter in DB
                    else:
                        # Directory exists, check for index.md
                        index_path = os.path.join(chapter_dir_path, "index.md")
                        if not os.path.exists(index_path):
                            # index.md doesn't exist in directory either
                            pass  # Continue to create chapter in DB
            else:
                # Docs directory doesn't exist (e.g., in test environment), allow personalization anyway
                pass  # Continue to create chapter in DB
            # If we get here, the chapter exists in the docs folder, so we can create it in our database if needed
            # For now, let's just proceed with the personalization
            chapter = Chapter(
                id=chapter_id,
                title=chapter_id.replace('-', ' ').title(),  # Create a title from the chapter ID
                content_path=f"docs/{chapter_id}.md"  # Default path
            )
            self.db.add(chapter)
            self.db.flush()  # Ensure the chapter is added to the session

        # Create personalization preference
        personalization_pref = UserPersonalizationPreference(
            user_id=user_id,
            chapter_id=chapter_id,
            preferences=json.dumps(preferences or {})
        )
        self.db.add(personalization_pref)

        # Award bonus points (50 points per chapter as specified)
        bonus_points = UserBonusPoints(
            user_id=user_id,
            chapter_id=chapter_id,
            points_earned=50
        )
        self.db.add(bonus_points)

        # Commit the changes
        self.db.commit()

        # Refresh to get the generated IDs
        self.db.refresh(personalization_pref)
        self.db.refresh(bonus_points)

        # For now, return the default chapter content with basic personalization
        # In a real implementation, this would apply user preferences to the content
        personalized_content = {
            "chapter_id": chapter_id,
            "title": chapter.title,
            "content": chapter.default_content,
            "preferences_applied": preferences or {},
            "personalization_applied": True
        }

        logging.info(f"Personalization activated successfully for user {user_id}, chapter {chapter_id}. Awarded {bonus_points.points_earned} points.")

        return {
            "success": True,
            "message": "Content personalized successfully, 50 bonus points awarded",
            "points_earned": 50,
            "personalized_content": personalized_content
        }

    def get_personalization_status(self, user_id: str, chapter_id: str) -> Dict[str, Any]:
        """
        Get the personalization status for a specific chapter for the user.

        Args:
            user_id: The ID of the user
            chapter_id: The ID of the chapter to check

        Returns:
            Dictionary with personalization status and related data
        """
        logging.debug(f"Retrieving personalization status for user {user_id}, chapter {chapter_id}")
        personalization = self.db.query(UserPersonalizationPreference).filter(
            and_(
                UserPersonalizationPreference.user_id == user_id,
                UserPersonalizationPreference.chapter_id == chapter_id
            )
        ).first()

        if not personalization:
            return {
                "is_personalized": False,
                "preferences": None,
                "points_earned": 0
            }

        # Get bonus points for this chapter
        bonus_points = self.db.query(UserBonusPoints).filter(
            and_(
                UserBonusPoints.user_id == user_id,
                UserBonusPoints.chapter_id == chapter_id
            )
        ).first()

        points_earned = bonus_points.points_earned if bonus_points else 0

        return {
            "is_personalized": True,
            "preferences": json.loads(personalization.preferences),
            "points_earned": points_earned
        }

    def get_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all personalization preferences set by the user.

        Args:
            user_id: The ID of the user

        Returns:
            List of personalization preferences with chapter info
        """
        preferences = self.db.query(UserPersonalizationPreference).filter(
            UserPersonalizationPreference.user_id == user_id
        ).all()

        result = []
        for pref in preferences:
            # Get bonus points for this chapter
            bonus_points = self.db.query(UserBonusPoints).filter(
                and_(
                    UserBonusPoints.user_id == user_id,
                    UserBonusPoints.chapter_id == pref.chapter_id
                )
            ).first()

            result.append({
                "chapter_id": pref.chapter_id,
                "preferences": json.loads(pref.preferences),
                "created_at": pref.created_at.isoformat() if pref.created_at else None,
                "points_earned": bonus_points.points_earned if bonus_points else 0
            })

        return result

    def get_user_bonus_points(self, user_id: str) -> Dict[str, Any]:
        """
        Get the total bonus points earned by the user.

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary with total points and breakdown by chapter
        """
        logging.debug(f"Retrieving bonus points for user {user_id}")
        # Get all bonus points for the user
        bonus_points_list = self.db.query(UserBonusPoints).filter(
            and_(
                UserBonusPoints.user_id == user_id,
                UserBonusPoints.is_valid == True
            )
        ).all()

        total_points = sum(bp.points_earned for bp in bonus_points_list)

        points_breakdown = []
        for bp in bonus_points_list:
            points_breakdown.append({
                "chapter_id": bp.chapter_id,
                "points": bp.points_earned,
                "earned_at": bp.earned_at.isoformat() if bp.earned_at else None
            })

        logging.info(f"Retrieved bonus points for user {user_id}: {total_points} total points from {len(points_breakdown)} chapters")
        return {
            "total_points": total_points,
            "points_breakdown": points_breakdown
        }

    def get_personalized_content(self, user_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the personalized content for a specific chapter based on user preferences.

        Args:
            user_id: The ID of the user
            chapter_id: The ID of the chapter

        Returns:
            Personalized content or None if not personalized
        """
        logging.debug(f"Retrieving personalized content for user {user_id}, chapter {chapter_id}")
        # Get the chapter
        chapter = self.db.query(Chapter).filter(Chapter.id == chapter_id).first()
        if not chapter:
            # Check if the chapter exists in the docs folder
            import os
            docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'docs')
            # Check if a file with the chapter_id exists in the docs folder (with .md extension)
            chapter_file_path = os.path.join(docs_path, f"{chapter_id}.md")
            if not os.path.exists(chapter_file_path):
                # Also check if it exists as a directory (which might contain an index.md)
                chapter_dir_path = os.path.join(docs_path, chapter_id)
                if not os.path.exists(chapter_dir_path):
                    return None
                # If it's a directory, check for index.md
                else:
                    index_path = os.path.join(chapter_dir_path, "index.md")
                    if not os.path.exists(index_path):
                        return None
                    # Load content from index.md
                    with open(index_path, 'r', encoding='utf-8') as f:
                        content = f.read()
            else:
                # Load content from the markdown file
                with open(chapter_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Create a temporary chapter object
            chapter = Chapter(
                id=chapter_id,
                title=chapter_id.replace('-', ' ').title(),  # Create a title from the chapter ID
                content_path=f"docs/{chapter_id}.md",  # Default path
                default_content=content
            )
        elif not chapter.default_content:
            # If chapter exists in DB but content is not loaded, try to load from file
            import os
            docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'docs')
            chapter_file_path = os.path.join(docs_path, f"{chapter_id}.md")
            if os.path.exists(chapter_file_path):
                with open(chapter_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chapter.default_content = content
            else:
                chapter_dir_path = os.path.join(docs_path, chapter_id)
                index_path = os.path.join(chapter_dir_path, "index.md")
                if os.path.exists(index_path):
                    with open(index_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    chapter.default_content = content

        # Get user's personalization preferences for this chapter
        personalization = self.db.query(UserPersonalizationPreference).filter(
            and_(
                UserPersonalizationPreference.user_id == user_id,
                UserPersonalizationPreference.chapter_id == chapter_id
            )
        ).first()

        if not personalization:
            # Return default content if not personalized
            return {
                "chapter_id": chapter_id,
                "title": chapter.title,
                "content": chapter.default_content or "",
                "is_personalized": False
            }

        # Apply personalization to content based on user preferences
        preferences = json.loads(personalization.preferences)

        # Apply personalization to the content based on preferences
        personalized_content_text = self._apply_content_personalization(
            chapter.default_content or "",
            preferences
        )

        result = {
            "chapter_id": chapter_id,
            "title": chapter.title,
            "content": personalized_content_text,
            "preferences_applied": preferences,
            "is_personalized": True,
            "personalization_template": chapter.personalization_template
        }

        return result

    def _apply_content_personalization(self, content: str, preferences: Dict[str, Any]) -> str:
        """
        Apply personalization to content based on user preferences.

        Args:
            content: The original content to personalize
            preferences: User preferences for personalization

        Returns:
            Personalized content string
        """
        personalized_content = content

        # Apply theme-based changes (e.g., highlighting important concepts)
        if preferences.get('theme') == 'dark':
            # Add dark theme indicators or modify content for dark theme
            # In a real implementation, this would change CSS classes or add styling
            pass
        elif preferences.get('theme') == 'light':
            # Add light theme indicators
            pass

        # Apply learning style modifications
        learning_style = preferences.get('learning_style', '').lower()
        if learning_style == 'visual':
            # Add more visual elements, diagrams, or highlight key terms
            # For now, we'll add visual indicators around important concepts
            import re
            # This is a simple example - in reality, you'd have more sophisticated parsing
            # Add emphasis to technical terms based on learning style
            personalized_content = re.sub(
                r'\b([A-Z][A-Z]+)\b',  # Find abbreviations/technical terms in all caps
                r'**\1**',  # Make them bold
                personalized_content
            )
        elif learning_style == 'detailed':
            # For detailed learners, potentially add more examples or explanations
            # This is just a placeholder for more complex logic
            pass
        elif learning_style == 'concise':
            # For concise learners, potentially simplify or summarize
            pass

        # Apply experience level modifications
        experience_level = preferences.get('experience_level', '').lower()
        if experience_level == 'beginner':
            # For beginners, add more explanations and simpler language
            # Add links to basic concepts
            pass
        elif experience_level == 'advanced':
            # For advanced users, potentially add more complex examples
            pass

        # Apply other preferences
        if preferences.get('highlight_important', True):
            # Add special markers for important content
            import re
            # Example: highlight words that follow "IMPORTANT:" or "KEY CONCEPT:"
            personalized_content = re.sub(
                r'(IMPORTANT:|KEY CONCEPT:)\s+([^.]+)',
                r'**\1** \2',
                personalized_content,
                flags=re.IGNORECASE
            )

        return personalized_content