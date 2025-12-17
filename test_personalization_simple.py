"""
Simple test for personalization functionality
Tests the personalization feature with acceptance scenario 1
"""
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models import Base, User, UserPersonalizationPreference, UserBonusPoints, Chapter
from backend.services.personalization_service import PersonalizationService

def test_personalization_functionality():
    """Test personalization functionality with acceptance scenario 1"""
    print("Testing personalization functionality...")

    # Create an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=True,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Create a test user
        test_user = User(
            user_id="test_user_123",
            email="test@example.com",
            hashed_password="test_hashed_password"
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

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        db.close()

def test_personalization_with_nonexistent_chapter():
    """Test personalization with a chapter that doesn't exist in the database but exists as a file"""
    print("\nTesting personalization with chapter that should be created dynamically...")

    # Create an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=True,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Create a test user
        test_user = User(
            user_id="test_user_456",
            email="test2@example.com",
            hashed_password="test_hashed_password"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        user_id = test_user.user_id
        # Use a chapter that doesn't exist in the database yet
        chapter_id = "dynamic-chapter-test"  # This will be created dynamically

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
            print("✓ Personalization works with dynamic chapter creation")

            # Check that the chapter was created in the database
            created_chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()
            if created_chapter:
                print("✓ Chapter was created in the database as expected")
            else:
                print("✗ Chapter was not created in the database")
                return False

            return True
        else:
            print("✗ Personalization failed with dynamic chapter")
            return False

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        db.close()

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