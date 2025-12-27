"""
Translation API endpoints for Urdu translation functionality.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from typing import Optional

from ..models.translation import (
    TranslateRequest,
    TranslateResponse,
    TranslationStatusRequest,
    TranslationStatusResponse,
    TranslationCacheRequest,
    TranslationCacheResponse
)
from ..services.translation_agent import translation_agent
from ..services.translation_cache import translation_cache
from ..middleware.translation_auth import verify_translation_access
from ..middleware.rate_limiter import check_rate_limit, get_rate_limit_headers


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
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"User {user_id} has translation permission")
    return True

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/translation", tags=["translation"])

# In-memory storage for translation status (in production, use a proper database)
translation_status_store = {}

@router.post("/translate", response_model=TranslateResponse)
async def translate_content(
    request: TranslateRequest,
    fastapi_request: Request,
    current_user: dict = Depends(verify_translation_access)
):
    """
    Translate content to Urdu while preserving formatting and structure.

    Args:
        request: The translation request containing content and metadata
        current_user: The authenticated user making the request

    Returns:
        TranslateResponse with the translated content
    """
    try:
        # Check rate limit before processing
        await check_rate_limit(fastapi_request, f"user:{current_user.get('id')}")

        # Verify user has permission to translate
        if not await user_can_translate(current_user):
            raise HTTPException(
                status_code=403,
                detail="User does not have permission to use translation service"
            )

        user_id = current_user.get("id")

        # Check if translation is already cached
        cached_result = await translation_cache.get_translation(user_id, request.chapter_id)
        if cached_result:
            logger.info(f"Returning cached translation for user {user_id}, chapter {request.chapter_id}")

            return TranslateResponse(
                translated_content=cached_result.content,
                chapter_id=request.chapter_id,
                user_id=user_id,
                translation_id=f"cached_{request.chapter_id}_{user_id}",
                created_at=datetime.fromtimestamp(cached_result.created_at)
            )

        # Generate a unique translation ID
        translation_id = f"tr_{uuid.uuid4().hex}"

        # Log the translation request
        logger.info(f"Starting translation for user {user_id}, chapter {request.chapter_id}, translation {translation_id}")

        # Perform the translation
        translated_content = await translation_agent.translate_content_preserve_formatting(
            content=request.content,
            user_id=user_id,
            chapter_id=request.chapter_id
        )

        if not translated_content:
            raise HTTPException(
                status_code=500,
                detail="Translation failed. Please try again later."
            )

        # Cache the translation
        await translation_cache.set_translation(
            user_id=user_id,
            chapter_id=request.chapter_id,
            content=translated_content
        )

        # Create and return response
        response = TranslateResponse(
            translated_content=translated_content,
            chapter_id=request.chapter_id,
            user_id=user_id,
            translation_id=translation_id,
            created_at=datetime.utcnow()
        )

        logger.info(f"Successfully translated content for user {user_id}, chapter {request.chapter_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in translate_content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation service error: {str(e)}"
        )


@router.post("/translate-async")
async def translate_content_async(
    request: TranslateRequest,
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_translation_access)
):
    """
    Asynchronously translate content to Urdu (for longer content).

    Args:
        request: The translation request containing content and metadata
        fastapi_request: The FastAPI request object
        background_tasks: FastAPI background tasks object
        current_user: The authenticated user making the request

    Returns:
        A response with translation ID for status checking
    """
    try:
        # Check rate limit before processing
        await check_rate_limit(fastapi_request, f"user:{current_user.get('id')}")

        # Verify user has permission to translate
        if not await user_can_translate(current_user):
            raise HTTPException(
                status_code=403,
                detail="User does not have permission to use translation service"
            )

        user_id = current_user.get("id")

        # Check if translation is already cached
        cached_result = await translation_cache.get_translation(user_id, request.chapter_id)
        if cached_result:
            logger.info(f"Returning cached translation for user {user_id}, chapter {request.chapter_id}")

            return {
                "translation_id": f"cached_{request.chapter_id}_{user_id}",
                "status": "completed",
                "cached": True,
                "chapter_id": request.chapter_id,
                "user_id": user_id
            }

        # Generate a unique translation ID
        translation_id = f"tr_{uuid.uuid4().hex}"

        # Store initial status as pending
        translation_status_store[translation_id] = {
            "status": "pending",
            "user_id": user_id,
            "chapter_id": request.chapter_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Add background task to perform translation
        background_tasks.add_task(
            _perform_async_translation,
            translation_id=translation_id,
            content=request.content,
            user_id=user_id,
            chapter_id=request.chapter_id
        )

        logger.info(f"Started async translation for user {user_id}, chapter {request.chapter_id}, translation {translation_id}")

        return {
            "translation_id": translation_id,
            "status": "pending",
            "chapter_id": request.chapter_id,
            "user_id": user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in translate_content_async: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation service error: {str(e)}"
        )


async def _perform_async_translation(translation_id: str, content: str, user_id: str, chapter_id: str):
    """
    Perform the actual translation in the background.

    Args:
        translation_id: The unique ID for this translation
        content: The content to translate
        user_id: The user ID requesting translation
        chapter_id: The chapter ID being translated
    """
    try:
        # Update status to in-progress
        translation_status_store[translation_id]["status"] = "in-progress"

        # Perform the translation
        translated_content = await translation_agent.translate_content_preserve_formatting(
            content=content,
            user_id=user_id,
            chapter_id=chapter_id
        )

        if translated_content:
            # Update status to completed and store result
            translation_status_store[translation_id].update({
                "status": "completed",
                "translated_content": translated_content,
                "completed_at": datetime.utcnow().isoformat()
            })

            # Cache the translation
            await translation_cache.set_translation(
                user_id=user_id,
                chapter_id=chapter_id,
                content=translated_content
            )
        else:
            # Mark as failed
            translation_status_store[translation_id]["status"] = "failed"
            translation_status_store[translation_id]["error"] = "Translation failed"

    except Exception as e:
        logger.error(f"Error in async translation {translation_id}: {str(e)}")
        translation_status_store[translation_id]["status"] = "failed"
        translation_status_store[translation_id]["error"] = str(e)


@router.post("/status", response_model=TranslationStatusResponse)
async def get_translation_status(
    request: TranslationStatusRequest,
    fastapi_request: Request,
    current_user: dict = Depends(verify_translation_access)
):
    """
    Get the status of an async translation.

    Args:
        request: The request containing the translation ID
        fastapi_request: The FastAPI request object
        current_user: The authenticated user making the request

    Returns:
        TranslationStatusResponse with current status
    """
    try:
        # Check rate limit before processing
        await check_rate_limit(fastapi_request, f"user:{current_user.get('id')}")

        user_id = current_user.get("id")
        translation_id = request.translation_id

        if translation_id not in translation_status_store:
            raise HTTPException(
                status_code=404,
                detail="Translation not found"
            )

        status_info = translation_status_store[translation_id]

        # Verify this translation belongs to the current user
        if status_info.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        return TranslationStatusResponse(
            translation_id=translation_id,
            status=status_info["status"],
            translated_content=status_info.get("translated_content"),
            error_message=status_info.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_translation_status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check error: {str(e)}"
        )


@router.post("/cache/check", response_model=TranslationCacheResponse)
async def check_translation_cache(
    request: TranslationCacheRequest,
    fastapi_request: Request,
    current_user: dict = Depends(verify_translation_access)
):
    """
    Check if a translation is already cached for a user and chapter.

    Args:
        request: The request containing user ID and chapter ID
        fastapi_request: The FastAPI request object
        current_user: The authenticated user making the request

    Returns:
        TranslationCacheResponse indicating if translation is cached
    """
    try:
        # Check rate limit before processing
        await check_rate_limit(fastapi_request, f"user:{current_user.get('id')}")

        # Verify the user making the request is the same as in the request
        requesting_user_id = current_user.get("id")
        if request.user_id and request.user_id != requesting_user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        # Use the authenticated user's ID if not provided in request
        user_id = requesting_user_id
        chapter_id = request.chapter_id

        # Check if translation is cached
        is_cached = await translation_cache.has_translation(user_id, chapter_id)

        cached_content = None
        if is_cached:
            cached_result = await translation_cache.get_translation(user_id, chapter_id)
            if cached_result:
                cached_content = cached_result.content

        return TranslationCacheResponse(
            is_cached=is_cached,
            cached_content=cached_content,
            chapter_id=chapter_id,
            user_id=user_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_translation_cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache check error: {str(e)}"
        )


@router.get("/health")
async def translation_health():
    """
    Health check endpoint for the translation service.

    Returns:
        Health status of the translation service
    """
    try:
        # Check if the translation agent is healthy
        is_healthy = await translation_agent.health_check()

        if is_healthy:
            return {"status": "healthy", "service": "translation"}
        else:
            return {"status": "unhealthy", "service": "translation"}

    except Exception as e:
        logger.error(f"Translation health check failed: {str(e)}")
        return {"status": "unhealthy", "service": "translation", "error": str(e)}


# Cleanup expired entries from status store periodically
async def cleanup_expired_statuses():
    """
    Cleanup function to remove expired translation status entries.
    This should be called periodically by a background task or scheduler.
    """
    current_time = datetime.utcnow().timestamp()
    expired_ids = []

    for translation_id, status_info in translation_status_store.items():
        # Consider entries older than 1 hour as expired
        created_at = datetime.fromisoformat(status_info["created_at"].replace("Z", "+00:00"))
        if (datetime.utcnow() - created_at).total_seconds() > 3600:
            expired_ids.append(translation_id)

    for translation_id in expired_ids:
        del translation_status_store[translation_id]

    logger.info(f"Cleaned up {len(expired_ids)} expired translation statuses")