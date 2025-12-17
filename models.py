from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # Unique identifier
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # This will reference our user ID
    email = Column(String, unique=True, index=True, nullable=False)  # Store email for reference
    software_experience = Column(String, nullable=False)  # beginner, intermediate, advanced
    hardware_experience = Column(String, nullable=False)  # none, basic, advanced
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UserPersonalizationPreference(Base):
    __tablename__ = "user_personalization_preferences"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)  # Reference to the user who owns this preference
    chapter_id = Column(String, nullable=False)  # Identifier for the chapter being personalized (based on existing docs folder structure)
    preferences = Column(Text, nullable=False)  # User-specific content customization settings (profile customizations applied to chapter)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Add unique constraint to prevent duplicate personalization for same user-chapter combination
    __table_args__ = (UniqueConstraint('user_id', 'chapter_id', name='unique_user_chapter_personalization'),)

    # Relationship
    user = relationship("User", back_populates="personalization_preferences")


# Add relationship to User model
User.personalization_preferences = relationship("UserPersonalizationPreference", order_by=UserPersonalizationPreference.id, back_populates="user")


class UserBonusPoints(Base):
    __tablename__ = "user_bonus_points"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)  # Reference to the user who earned these points
    chapter_id = Column(String, nullable=False)  # Identifier for the chapter where points were earned
    points_earned = Column(Integer, nullable=False, default=50)  # Number of points earned (50 per chapter as specified)
    earned_at = Column(DateTime(timezone=True), server_default=func.now())
    is_valid = Column(Boolean, nullable=False, default=True)  # Whether these points are still valid (to handle potential abuse)

    # Add unique constraint to prevent duplicate points for same user-chapter combination
    __table_args__ = (UniqueConstraint('user_id', 'chapter_id', name='unique_user_chapter_bonus_points'),)

    # Relationship
    user = relationship("User", back_populates="bonus_points")


# Add relationship to User model
User.bonus_points = relationship("UserBonusPoints", order_by=UserBonusPoints.id, back_populates="user")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(String, primary_key=True, nullable=False)  # Unique identifier for the chapter (based on existing docs folder structure)
    title = Column(String, nullable=False)  # Chapter title
    content_path = Column(String, nullable=False)  # File path to the chapter content in the docs folder
    default_content = Column(Text)  # Default chapter content before personalization
    personalization_template = Column(Text)  # Template for how content can be personalized using user profile customizations
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Note: Relationships to personalization and bonus points are handled in the service layer
    # since chapter_id is a string that may reference docs files without strict foreign key constraints


# Note: We don't define relationships to Chapter with foreign keys since chapter_id is a string that may reference docs files
# The personalization service handles the relationship logic in code
