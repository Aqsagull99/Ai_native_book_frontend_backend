"""
Authentication middleware to verify user session for translation access.
"""
import logging
from typing import Optional
from fastapi import Request, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from auth import verify_token

logger = logging.getLogger(__name__)

async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Extract the current user from the request by verifying the JWT token.
    This follows the same pattern as the personalization endpoints.
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


async def verify_translation_access(request: Request, db: AsyncSession = Depends(get_db)) -> Optional[dict]:
    """
    Verify that the user has a valid session and can access translation functionality.

    Args:
        request: The incoming request object
        db: Database session

    Returns:
        User object if authenticated, None otherwise

    Raises:
        HTTPException: If user is not authenticated
    """
    try:
        # Use the same authentication method as personalization endpoints
        current_user = await get_current_user(request, db)

        if not current_user:
            logger.warning(f"Unauthorized translation access attempt from IP: {request.client.host}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for translation service"
            )

        # Verify user account is active and not suspended
        user_data = current_user
        if user_data.get("disabled", False):
            logger.warning(f"Access denied for disabled user: {user_data.get('id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled"
            )

        logger.info(f"User {user_data.get('id')} authenticated for translation access")
        return user_data

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error verifying translation access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for translation service"
        )


async def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Extract and return the user ID from the request if authenticated.

    Args:
        request: The incoming request object

    Returns:
        User ID if authenticated, None otherwise
    """
    try:
        user = await verify_translation_access(request)
        return user.get("id") if user else None
    except HTTPException:
        # Return None if not authenticated
        return None


def require_translation_auth():
    """
    Decorator-like function that can be used as a dependency to require
    authentication for translation endpoints.
    """
    async def auth_dependency(request: Request):
        return await verify_translation_access(request)

    return auth_dependency


# Additional utility functions for checking user permissions
async def user_can_translate(user: dict) -> bool:
    """
    Check if a user has permission to use the translation service.
    This can be extended to check user roles, subscription status, etc.

    Args:
        user: The authenticated user object

    Returns:
        True if user can translate, False otherwise
    """
    if not user:
        return False

    # Basic check: user must be authenticated
    # This can be extended to check roles, permissions, subscription status, etc.
    user_id = user.get("id")
    email = user.get("email")

    if not user_id:
        return False

    # For now, any authenticated user can translate
    # This could be enhanced with role-based permissions
    logger.debug(f"User {user_id} has translation permission")
    return True