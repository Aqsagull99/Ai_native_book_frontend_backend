"""
Integration test for the complete personalization feature
Tests all user stories and acceptance criteria together
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models import Base, User, UserPersonalizationPreference, UserBonusPoints, Chapter
from backend.services.personalization_service import PersonalizationService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

def test_complete_personalization_workflow():
    """Test the complete personalization workflow"""
    print("Testing complete personalization workflow...")

    # Create an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
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
            user_id="integration_test_user_123",
            email="integration@example.com",
            hashed_password="test_hashed_password"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        user_id = test_user.user_id

        print(f"Created test user with ID: {user_id}")

        # Initialize the personalization service
        service = PersonalizationService(db)

        # Test User Story 1: Personalize Chapter Content
        print("\n--- Testing User Story 1: Personalize Chapter Content ---")

        # Test 1: Activate personalization for first chapter
        chapter_id_1 = "integration-test-chapter-1"
        result1 = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id_1,
            preferences={"theme": "dark", "learning_style": "visual"}
        )

        if result1["success"] and result1["points_earned"] == 50:
            print("✓ User Story 1 - Personalization activation successful (50 points awarded)")
        else:
            print("✗ User Story 1 - Personalization activation failed")
            return False

        # Test 2: Try to activate same chapter again (should fail - duplicate protection)
        result_duplicate = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id_1,
            preferences={"theme": "light"}
        )

        if not result_duplicate["success"] and result_duplicate.get("is_duplicate"):
            print("✓ User Story 1 - Duplicate protection working correctly")
        else:
            print("✗ User Story 1 - Duplicate protection failed")
            return False

        # Test User Story 2: View Personalized Content
        print("\n--- Testing User Story 2: View Personalized Content ---")

        # Get personalized content
        content_result = service.get_personalized_content(user_id=user_id, chapter_id=chapter_id_1)

        if content_result and content_result["is_personalized"]:
            print("✓ User Story 2 - Personalized content retrieved successfully")
        else:
            print("✗ User Story 2 - Failed to retrieve personalized content")
            return False

        # Test User Story 3: Track Bonus Points
        print("\n--- Testing User Story 3: Track Bonus Points ---")

        # Get user bonus points
        points_result = service.get_user_bonus_points(user_id=user_id)

        if points_result["total_points"] == 50 and len(points_result["points_breakdown"]) == 1:
            print("✓ User Story 3 - Bonus points tracking working correctly")
        else:
            print("✗ User Story 3 - Bonus points tracking failed")
            return False

        # Test multiple chapters
        print("\n--- Testing Multiple Chapters ---")

        # Personalize second chapter
        chapter_id_2 = "integration-test-chapter-2"
        result2 = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id_2,
            preferences={"theme": "light", "fontSize": "large"}
        )

        if result2["success"] and result2["points_earned"] == 50:
            print("✓ Multiple chapters - Second chapter personalized successfully")
        else:
            print("✗ Multiple chapters - Second chapter personalization failed")
            return False

        # Check total points after second chapter
        total_points = service.get_user_bonus_points(user_id=user_id)

        if total_points["total_points"] == 100 and len(total_points["points_breakdown"]) == 2:
            print("✓ Multiple chapters - Total points accumulated correctly (100 points)")
        else:
            print("✗ Multiple chapters - Points accumulation failed")
            return False

        # Test status checking
        print("\n--- Testing Status Checking ---")

        status1 = service.get_personalization_status(user_id=user_id, chapter_id=chapter_id_1)
        status2 = service.get_personalization_status(user_id=user_id, chapter_id=chapter_id_2)

        if status1["is_personalized"] and status2["is_personalized"]:
            print("✓ Status checking - Both chapters show as personalized")
        else:
            print("✗ Status checking - Status not correct")
            return False

        # Test preferences retrieval
        print("\n--- Testing Preferences Retrieval ---")

        user_preferences = service.get_user_preferences(user_id=user_id)

        if len(user_preferences) == 2:
            print("✓ Preferences retrieval - All preferences retrieved correctly")
        else:
            print("✗ Preferences retrieval - Incorrect number of preferences")
            return False

        print("\n✓ All user stories and acceptance criteria passed!")
        print(f"✓ Total bonus points earned: {total_points['total_points']}")
        print(f"✓ Chapters personalized: {len(total_points['points_breakdown'])}")
        print(f"✓ User has {len(user_preferences)} personalization preferences saved")

        return True

    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        db.close()

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n--- Testing Edge Cases ---")

    # Create an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
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
            user_id="edge_case_test_user_456",
            email="edge@example.com",
            hashed_password="test_hashed_password"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        user_id = test_user.user_id
        service = PersonalizationService(db)

        # Test with non-existent chapter (should create it dynamically)
        chapter_id = "non-existent-chapter-test"
        result = service.activate_personalization(
            user_id=user_id,
            chapter_id=chapter_id,
            preferences={"theme": "dark"}
        )

        if result["success"]:
            print("✓ Edge case - Non-existent chapter handled gracefully")
        else:
            print("✗ Edge case - Non-existent chapter not handled properly")
            return False

        # Test invalid user (this would be caught at the API level, not service level)
        print("✓ Edge cases passed")
        return True

    except Exception as e:
        print(f"Error during edge case test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        db.close()

if __name__ == "__main__":
    print("Starting integration tests for personalization feature...")

    success1 = test_complete_personalization_workflow()
    success2 = test_edge_cases()

    if success1 and success2:
        print("\n🎉 All integration tests passed! Feature is working correctly.")
        print("\nSummary of tested functionality:")
        print("- User Story 1: Personalize Chapter Content ✓")
        print("- User Story 2: View Personalized Content ✓")
        print("- User Story 3: Track Bonus Points ✓")
        print("- Duplicate protection ✓")
        print("- Multiple chapter support ✓")
        print("- Status checking ✓")
        print("- Preferences management ✓")
        print("- Edge case handling ✓")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed!")
        sys.exit(1)