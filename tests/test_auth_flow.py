# Test script to verify the complete authentication flow
import asyncio
import httpx
import json

async def test_auth_flow():
    base_url = "http://localhost:8000"

    print("Testing complete authentication flow...")

    # Test signup
    print("\n1. Testing signup...")
    signup_data = {
        "email": "test@example.com",
        "password": "securePassword123",
        "software_experience": "beginner",
        "hardware_experience": "none"
    }

    async with httpx.AsyncClient() as client:
        # Signup
        response = await client.post(f"{base_url}/api/auth/signup", json=signup_data)
        print(f"Signup response: {response.status_code}")
        if response.status_code == 200:
            print("✓ Signup successful")
            result = response.json()
            print(f"  User ID: {result.get('user_id')}")
        else:
            print(f"✗ Signup failed: {response.text}")
            return False

        # Signin
        print("\n2. Testing signin...")
        signin_data = {
            "email": "test@example.com",
            "password": "securePassword123"
        }

        response = await client.post(f"{base_url}/api/auth/signin", json=signin_data)
        print(f"Signin response: {response.status_code}")
        if response.status_code == 200:
            print("✓ Signin successful")
        else:
            print(f"✗ Signin failed: {response.text}")
            return False

        # Get profile
        print("\n3. Testing profile retrieval...")
        response = await client.get(f"{base_url}/api/auth/profile?email=test@example.com")
        print(f"Profile response: {response.status_code}")
        if response.status_code == 200:
            profile = response.json()
            print(f"✓ Profile retrieved: {profile}")
        else:
            print(f"✗ Profile retrieval failed: {response.text}")
            return False

        # Update profile
        print("\n4. Testing profile update...")
        update_data = {
            "software_experience": "intermediate",
            "hardware_experience": "basic"
        }

        response = await client.put(f"{base_url}/api/auth/profile?email=test@example.com", json=update_data)
        print(f"Profile update response: {response.status_code}")
        if response.status_code == 200:
            print("✓ Profile updated successfully")
        else:
            print(f"✗ Profile update failed: {response.text}")
            return False

    print("\n✓ All tests passed! Complete authentication flow is working.")
    return True

if __name__ == "__main__":
    asyncio.run(test_auth_flow())