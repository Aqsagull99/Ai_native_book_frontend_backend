"""
Personalization API Endpoints
Handles all personalization-related API requests
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from database import get_db
from services.personalization_service import PersonalizationService
from pydantic import BaseModel

# Router for personalization-specific endpoints
personalization_router = APIRouter(prefix="/personalization", tags=["personalization"])

# Router for user-related endpoints (bonus points, etc.)
user_router = APIRouter(prefix="/user", tags=["user"])

# Pydantic model for activate personalization request
class ActivatePersonalizationRequest(BaseModel):
    chapter_id: str
    preferences: Optional[Dict[str, Any]] = None

from auth import verify_token

async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Extract the current user from the request by verifying the JWT token.
    """
    # Extract the Authorization header
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")

    token = auth_header[7:]  # Remove "Bearer " prefix
    # Decode and validate the JWT token
    payload = verify_token(token)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: no user ID")

    # Get user info from the database
    from models import User
    result = await db.execute(User.__table__.select().where(User.user_id == user_id))
    user = result.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {"user_id": user.user_id, "id": user.user_id, "email": user.email}


@personalization_router.post("/activate")
async def activate_personalization(
    personalization_request: ActivatePersonalizationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Activate personalization for a chapter and award bonus points.

    Args:
        personalization_request: Request body containing chapter_id and preferences
        request: The HTTP request object
        db: Database session
        current_user: The currently authenticated user

    Returns:
        Dictionary with success status, message, points earned, and personalized content
    """
    try:
        # Validate that the user is authenticated
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Extract data from the request model
        chapter_id = personalization_request.chapter_id
        preferences = personalization_request.preferences

        # Import models here to avoid circular imports
        from models import UserPersonalizationPreference, UserBonusPoints, Chapter
        from sqlalchemy import and_
        import json
        from datetime import datetime

        # Check if personalization already exists for this user-chapter combination
        existing_personalization_result = await db.execute(
            UserPersonalizationPreference.__table__.select().where(
                and_(
                    UserPersonalizationPreference.user_id == (current_user.get("user_id") or current_user.get("id")),
                    UserPersonalizationPreference.chapter_id == chapter_id
                )
            )
        )
        existing_personalization = existing_personalization_result.fetchone()

        if existing_personalization:
            return {
                "success": False,
                "message": "Personalization already activated for this chapter",
                "points_earned": 0,
                "is_duplicate": True
            }

        # Check if chapter exists
        chapter_result = await db.execute(
            Chapter.__table__.select().where(Chapter.id == chapter_id)
        )
        chapter = chapter_result.fetchone()

        if not chapter:
            # Create chapter if it doesn't exist
            from sqlalchemy.dialects.postgresql import insert
            from sqlalchemy import text
            import uuid

            new_chapter_id = str(uuid.uuid4())
            await db.execute(
                Chapter.__table__.insert().values(
                    id=chapter_id,
                    title=chapter_id.replace('-', ' ').title(),
                    content_path=f"docs/{chapter_id}.md",
                    created_at=datetime.utcnow()
                )
            )
            await db.commit()

        # Create personalization preference
        from sqlalchemy.dialects.postgresql import insert
        import uuid

        pref_id = str(uuid.uuid4())
        await db.execute(
            UserPersonalizationPreference.__table__.insert().values(
                id=pref_id,
                user_id=(current_user.get("user_id") or current_user.get("id")),
                chapter_id=chapter_id,
                preferences=json.dumps(preferences or {}),
                created_at=datetime.utcnow()
            )
        )

        # Award bonus points (50 points per chapter as specified)
        bonus_id = str(uuid.uuid4())
        await db.execute(
            UserBonusPoints.__table__.insert().values(
                id=bonus_id,
                user_id=(current_user.get("user_id") or current_user.get("id")),
                chapter_id=chapter_id,
                points_earned=50,
                earned_at=datetime.utcnow(),
                is_valid=True
            )
        )

        # Commit the changes
        await db.commit()

        # For now, return a basic response
        # In a real implementation, this would apply user preferences to the content
        personalized_content = {
            "chapter_id": chapter_id,
            "title": chapter_id.replace('-', ' ').title() if chapter else chapter_id,
            "preferences_applied": preferences or {},
            "personalization_applied": True
        }

        return {
            "success": True,
            "message": "Content personalized successfully, 50 bonus points awarded",
            "points_earned": 50,
            "personalized_content": personalized_content
        }
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except Exception as e:
        # Log the error and return a generic error message
        print(f"Error in activate_personalization: {str(e)}")  # In production, use proper logging
        raise HTTPException(status_code=500, detail="Internal server error")


@personalization_router.get("/status")
async def get_personalization_status(
    chapter_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the personalization status for a specific chapter for the user.

    Args:
        chapter_id: The ID of the chapter to check
        db: Database session
        current_user: The currently authenticated user

    Returns:
        Dictionary with personalization status and related data
    """
    try:
        # Validate that the user is authenticated
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Import models here to avoid circular imports
        from models import UserPersonalizationPreference, UserBonusPoints
        from sqlalchemy import and_
        import json

        # Get personalization status
        personalization_result = await db.execute(
            UserPersonalizationPreference.__table__.select().where(
                and_(
                    UserPersonalizationPreference.user_id == (current_user.get("user_id") or current_user.get("id")),
                    UserPersonalizationPreference.chapter_id == chapter_id
                )
            )
        )
        personalization = personalization_result.fetchone()

        if not personalization:
            return {
                "is_personalized": False,
                "preferences": None,
                "points_earned": 0
            }

        # Get bonus points for this chapter
        bonus_result = await db.execute(
            UserBonusPoints.__table__.select().where(
                and_(
                    UserBonusPoints.user_id == (current_user.get("user_id") or current_user.get("id")),
                    UserBonusPoints.chapter_id == chapter_id
                )
            )
        )
        bonus_points = bonus_result.fetchone()

        points_earned = bonus_points.points_earned if bonus_points else 0

        return {
            "is_personalized": True,
            "preferences": json.loads(personalization.preferences) if personalization.preferences else {},
            "points_earned": points_earned
        }
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except Exception as e:
        # Log the error and return a generic error message
        print(f"Error in get_personalization_status: {str(e)}")  # In production, use proper logging
        raise HTTPException(status_code=500, detail="Internal server error")


@personalization_router.get("/preferences")
async def get_user_preferences(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all personalization preferences set by the user.

    Args:
        db: Database session
        current_user: The currently authenticated user

    Returns:
        List of personalization preferences with chapter info
    """
    try:
        # Validate that the user is authenticated
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Import models here to avoid circular imports
        from models import UserPersonalizationPreference, UserBonusPoints
        from sqlalchemy import and_
        import json

        # Get all personalization preferences for the user
        preferences_result = await db.execute(
            UserPersonalizationPreference.__table__.select().where(
                UserPersonalizationPreference.user_id == (current_user.get("user_id") or current_user.get("id"))
            )
        )
        preferences = preferences_result.fetchall()

        result = []
        for pref in preferences:
            # Get bonus points for this chapter
            bonus_result = await db.execute(
                UserBonusPoints.__table__.select().where(
                    and_(
                        UserBonusPoints.user_id == (current_user.get("user_id") or current_user.get("id")),
                        UserBonusPoints.chapter_id == pref.chapter_id
                    )
                )
            )
            bonus_points = bonus_result.fetchone()

            result.append({
                "chapter_id": pref.chapter_id,
                "preferences": json.loads(pref.preferences) if pref.preferences else {},
                "created_at": pref.created_at.isoformat() if pref.created_at else None,
                "points_earned": bonus_points.points_earned if bonus_points else 0
            })

        return {"preferences": result}
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except Exception as e:
        # Log the error and return a generic error message
        print(f"Error in get_user_preferences: {str(e)}")  # In production, use proper logging
        raise HTTPException(status_code=500, detail="Internal server error")


@user_router.get("/bonus-points")
async def get_user_bonus_points(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the total bonus points earned by the user.

    Args:
        db: Database session
        current_user: The currently authenticated user

    Returns:
        Dictionary with total points and breakdown by chapter
    """
    try:
        # Validate that the user is authenticated
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Import models here to avoid circular imports
        from models import UserBonusPoints
        from sqlalchemy import and_

        # Get all bonus points for the user that are valid
        bonus_points_result = await db.execute(
            UserBonusPoints.__table__.select().where(
                and_(
                    UserBonusPoints.user_id == (current_user.get("user_id") or current_user.get("id")),
                    UserBonusPoints.is_valid == True
                )
            )
        )
        bonus_points_list = bonus_points_result.fetchall()

        total_points = sum(bp.points_earned for bp in bonus_points_list)

        points_breakdown = []
        for bp in bonus_points_list:
            points_breakdown.append({
                "chapter_id": bp.chapter_id,
                "points": bp.points_earned,
                "earned_at": bp.earned_at.isoformat() if bp.earned_at else None
            })

        return {
            "total_points": total_points,
            "points_breakdown": points_breakdown
        }
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper error codes
        raise
    except Exception as e:
        # Log the error and return a generic error message
        print(f"Error in get_user_bonus_points: {str(e)}")  # In production, use proper logging
        raise HTTPException(status_code=500, detail="Internal server error")


# Export both routers
router = personalization_router  # Keep the original name for backward compatibility
bonus_points_router = user_router  # New router for bonus points