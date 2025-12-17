import pytest
from fastapi.testclient import TestClient
from backend.main import app
from unittest.mock import AsyncMock, patch

# Create test client
client = TestClient(app)

def test_signup():
    """Test user signup endpoint"""
    with patch('backend.auth.create_better_auth_user') as mock_create_user:
        mock_create_user.return_value = AsyncMock()
        mock_create_user.return_value.id = "test_user_id"
        mock_create_user.return_value.email = "test@example.com"
        mock_create_user.return_value.emailVerified = True

        response = client.post(
            "/api/auth/signup",
            json={
                "email": "test@example.com",
                "password": "securepassword123",
                "software_experience": "beginner",
                "hardware_experience": "none"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "User registered successfully"

def test_signin():
    """Test user signin endpoint"""
    with patch('backend.auth.authenticate_better_auth_user') as mock_auth_user:
        mock_auth_user.return_value = AsyncMock()
        mock_auth_user.return_value.id = "test_user_id"
        mock_auth_user.return_value.email = "test@example.com"
        mock_auth_user.return_value.emailVerified = True

        # First, mock a signup to create a profile
        with patch('backend.auth.create_better_auth_user') as mock_create_user:
            mock_create_user.return_value = AsyncMock()
            mock_create_user.return_value.id = "test_user_id"
            mock_create_user.return_value.email = "test2@example.com"
            mock_create_user.return_value.emailVerified = True

            signup_response = client.post(
                "/api/auth/signup",
                json={
                    "email": "test2@example.com",
                    "password": "securepassword123",
                    "software_experience": "intermediate",
                    "hardware_experience": "basic"
                }
            )
            assert signup_response.status_code == 200

        # Then try to sign in
        response = client.post(
            "/api/auth/signin",
            json={
                "email": "test2@example.com",
                "password": "securepassword123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "User authenticated successfully"
        assert "email" in data

def test_get_profile():
    """Test get user profile endpoint"""
    # First, mock a signup to create a profile
    with patch('backend.auth.create_better_auth_user') as mock_create_user:
        mock_create_user.return_value = AsyncMock()
        mock_create_user.return_value.id = "profile_test_user_id"
        mock_create_user.return_value.email = "profiletest@example.com"
        mock_create_user.return_value.emailVerified = True

        signup_response = client.post(
            "/api/auth/signup",
            json={
                "email": "profiletest@example.com",
                "password": "securepassword123",
                "software_experience": "advanced",
                "hardware_experience": "advanced"
            }
        )
        assert signup_response.status_code == 200

    # Then try to get profile
    response = client.get("/api/auth/profile?email=profiletest@example.com")
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert data["email"] == "profiletest@example.com"

def test_update_profile():
    """Test update user profile endpoint"""
    # First, mock a signup to create a profile
    with patch('backend.auth.create_better_auth_user') as mock_create_user:
        mock_create_user.return_value = AsyncMock()
        mock_create_user.return_value.id = "update_test_user_id"
        mock_create_user.return_value.email = "updatetest@example.com"
        mock_create_user.return_value.emailVerified = True

        signup_response = client.post(
            "/api/auth/signup",
            json={
                "email": "updatetest@example.com",
                "password": "securepassword123",
                "software_experience": "beginner",
                "hardware_experience": "none"
            }
        )
        assert signup_response.status_code == 200

    # Then try to update profile
    response = client.put(
        "/api/auth/profile?email=updatetest@example.com",
        json={
            "software_experience": "intermediate",
            "hardware_experience": "basic"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Profile updated successfully"

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"