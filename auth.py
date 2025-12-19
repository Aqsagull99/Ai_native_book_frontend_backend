from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import User, UserProfile
import uuid
import re
from dotenv import load_dotenv
import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
import secrets
from jose import JWTError, jwt

# Load environment variables
load_dotenv()

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter()

# Pydantic models for request/response validation
class SignupRequest(BaseModel):
    email: str
    password: str
    software_experience: str  # beginner, intermediate, advanced
    hardware_experience: str  # none, basic, advanced


class SigninRequest(BaseModel):
    email: str
    password: str


class ProfileResponse(BaseModel):
    id: int
    user_id: str
    email: str
    software_experience: str
    hardware_experience: str

    class Config:
        from_attributes = True


class ProfileUpdateRequest(BaseModel):
    software_experience: Optional[str] = None  # beginner, intermediate, advanced
    hardware_experience: Optional[str] = None  # none, basic, advanced


class BetterAuthUserResponse(BaseModel):
    id: str
    email: str
    emailVerified: bool
    image: Optional[str] = None




@router.post("/signup")
async def signup(request: SignupRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user with experience levels.
    """
    try:
        # Input validation
        if request.software_experience not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid software experience level. Must be 'beginner', 'intermediate', or 'advanced'."
            )

        if request.hardware_experience not in ["none", "basic", "advanced"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid hardware experience level. Must be 'none', 'basic', or 'advanced'."
            )

        # Validate email format using a simple regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, request.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format."
            )

        # Check if user profile already exists in our database
        existing_profile = await db.execute(
            UserProfile.__table__.select().where(UserProfile.email == request.email)
        )
        existing_profile = existing_profile.fetchone()

        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )

        # Generate a unique user ID
        user_id = str(uuid.uuid4())

        # Check if user already exists in the users table
        existing_user = await db.execute(
            User.__table__.select().where(User.email == request.email)
        )
        existing_user = existing_user.fetchone()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )

        # Hash the password
        hashed_password = get_password_hash(request.password)

        # Create user in the users table
        user = User(
            user_id=user_id,
            email=request.email,
            hashed_password=hashed_password
        )

        # Create user profile in the user_profiles table
        user_profile = UserProfile(
            user_id=user_id,
            email=request.email,
            software_experience=request.software_experience,
            hardware_experience=request.hardware_experience
        )

        db.add(user)
        db.add(user_profile)
        await db.commit()
        await db.refresh(user)
        await db.refresh(user_profile)

        return {
            "message": "User registered successfully",
            "profile_id": user_profile.id,
            "user_id": user_profile.user_id,
            "email": user_profile.email
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering user: {str(e)}"
        )


@router.post("/signin")
async def signin(request: SigninRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate user.
    """
    try:
        # Validate email format using a simple regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, request.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format."
            )

        # Find the user in the database
        result = await db.execute(
            User.__table__.select().where(User.email == request.email)
        )
        user = result.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        # Verify the password
        if not verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        # Check if user profile exists in our database
        result = await db.execute(
            UserProfile.__table__.select().where(UserProfile.email == request.email)
        )
        user_profile = result.fetchone()

        if not user_profile:
            # If no profile exists but user is authenticated, this shouldn't happen in this implementation
            # since we create profile on signup, but we'll handle it gracefully
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_profile.user_id}, expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "message": "User authenticated successfully",
            "email": request.email,
            "user_id": user_profile.user_id
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error authenticating user: {str(e)}"
        )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Retrieve user profile for personalization.
    In a real implementation, the user ID would come from an auth token in the request header.
    For this implementation, we'll use a query parameter.
    """
    try:
        # In a real implementation, user_id would come from the auth token in the Authorization header
        # For this demo, we'll look up by email from query parameter
        email = request.query_params.get("email")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email parameter required"
            )

        # Query for the user profile
        result = await db.execute(
            UserProfile.__table__.select().where(UserProfile.email == email)
        )
        user_profile = result.fetchone()

        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Convert row to dict for response
        profile_dict = {
            "id": user_profile.id,
            "user_id": user_profile.user_id,
            "email": user_profile.email,
            "software_experience": user_profile.software_experience,
            "hardware_experience": user_profile.hardware_experience
        }

        return profile_dict
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving profile: {str(e)}"
        )


@router.put("/profile")
async def update_profile(request: Request, update_request: ProfileUpdateRequest, db: AsyncSession = Depends(get_db)):
    """
    Update user profile.
    In a real implementation, the user ID would come from an auth token in the request header.
    For this implementation, we'll use a query parameter.
    """
    try:
        # In a real implementation, user_id would come from the auth token in the Authorization header
        # For this demo, we'll use email from query parameter
        email = request.query_params.get("email")

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email parameter required"
            )

        # Get the user profile by email
        result = await db.execute(
            UserProfile.__table__.select().where(UserProfile.email == email)
        )
        user_profile = result.fetchone()

        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Prepare update values
        update_values = {}
        if update_request.software_experience is not None:
            update_values["software_experience"] = update_request.software_experience
        if update_request.hardware_experience is not None:
            update_values["hardware_experience"] = update_request.hardware_experience

        # Perform the update
        await db.execute(
            UserProfile.__table__.update()
            .where(UserProfile.email == email)
            .values(**update_values)
        )
        await db.commit()

        # Fetch the updated profile
        result = await db.execute(
            UserProfile.__table__.select().where(UserProfile.email == email)
        )
        updated_profile = result.fetchone()

        return {
            "message": "Profile updated successfully",
            "profile_id": updated_profile.id
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating profile: {str(e)}"
        )
