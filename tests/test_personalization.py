"""
Test script for personalization functionality
Tests the personalization feature with acceptance scenario 1
"""
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from database import engine, get_db
from models import User, UserPersonalizationPreference, UserBonusPoints, Chapter
from backend.services.personalization_service import PersonalizationService
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def test_personalization_functionality():
    """Test personalization functionality with acceptance scenario 1"""
    print("Testing personalization functionality...")

    # Create a test database session
    sync_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with sync_session() as db:
        # Create a test user if one doesn't exist
        test_user = db.query(User).first()
        if not test_user:
            test_user = User(
                email="test@example.com",
                hashed_password="test_hashed_password",
                full_name="Test User"
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)

        user_id = test_user.user_id
        chapter_id = "test-chapter-1"

        print(f"Using user_id: {user_id}")

        # Initialize the personalization service
        service = PersonalizationService(db)

        # Test 1: Activate personalization for the first time
        print("\nTest 1: Activating personalization for the first time")
        result1 = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id,
            preferences={"theme": "dark", "fontSize": "large"}
        )

        print(f"Result 1: {result1}")

        if result1["success"] and result1["points_earned"] == 50:
            print("✓ Personalization activated successfully and 50 points awarded")
        else:
            print("✗ Personalization activation failed")
            return False

        # Test 2: Try to activate personalization again (should fail as duplicate)
        print("\nTest 2: Attempting to activate personalization again (should fail as duplicate)")
        result2 = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id,
            preferences={"theme": "light", "fontSize": "small"}
        )

        print(f"Result 2: {result2}")

        if not result2["success"] and result2.get("is_duplicate"):
            print("✓ Duplicate activation correctly prevented")
        else:
            print("✗ Duplicate activation was not prevented")
            return False

        # Test 3: Check personalization status
        print("\nTest 3: Checking personalization status")
        status = service.get_personalization_status(user_id=user_id, chapter_id=chapter_id)
        print(f"Status: {status}")

        if status["is_personalized"] and status["points_earned"] == 50:
            print("✓ Personalization status is correct")
        else:
            print("✗ Personalization status is incorrect")
            return False

        # Test 4: Check user bonus points
        print("\nTest 4: Checking user bonus points")
        points = service.get_user_bonus_points(user_id=user_id)
        print(f"Points: {points}")

        if points["total_points"] == 50 and len(points["points_breakdown"]) == 1:
            print("✓ Bonus points tracking is correct")
        else:
            print("✗ Bonus points tracking is incorrect")
            return False

        # Test 5: Test with a different chapter
        print("\nTest 5: Testing with a different chapter")
        chapter_id_2 = "test-chapter-2"
        result3 = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id_2,
            preferences={"theme": "light", "fontSize": "medium"}
        )

        print(f"Result 3: {result3}")

        if result3["success"] and result3["points_earned"] == 50:
            print("✓ Personalization activated successfully for second chapter")
        else:
            print("✗ Personalization failed for second chapter")
            return False

        # Check total points after second chapter
        points_after_second = service.get_user_bonus_points(user_id=user_id)
        print(f"Points after second chapter: {points_after_second}")

        if points_after_second["total_points"] == 100 and len(points_after_second["points_breakdown"]) == 2:
            print("✓ Total bonus points correctly accumulated")
        else:
            print("✗ Total bonus points not correctly accumulated")
            return False

        print("\n✓ All tests passed! Personalization functionality works as expected.")
        return True

def test_personalization_with_nonexistent_chapter():
    """Test personalization with a chapter that doesn't exist in the database but exists as a file"""
    print("\nTesting personalization with chapter that exists as file...")

    # Create a test database session
    sync_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with sync_session() as db:
        # Create a test user if one doesn't exist
        test_user = db.query(User).first()
        if not test_user:
            test_user = User(
                email="test2@example.com",
                hashed_password="test_hashed_password",
                full_name="Test User 2"
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)

        user_id = test_user.user_id
        # Use an actual chapter from the docs folder
        chapter_id = "chapter-1-advanced-perception-training"  # This exists in frontend/docs

        # Initialize the personalization service
        service = PersonalizationService(db)

        print(f"Testing with user_id: {user_id}, chapter_id: {chapter_id}")

        result = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id,
            preferences={"theme": "dark", "learning_style": "visual"}
        )

        print(f"Result: {result}")

        if result["success"] and result["points_earned"] == 50:
            print("✓ Personalization works with chapters from docs folder")
            return True
        else:
            print("✗ Personalization failed with docs folder chapter")
            return False

if __name__ == "__main__":
    print("Starting personalization functionality tests...")

    success1 = test_personalization_functionality()
    success2 = test_personalization_with_nonexistent_chapter()

    if success1 and success2:
        print("\n✓ All personalization tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)