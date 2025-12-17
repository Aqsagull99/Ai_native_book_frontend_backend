#!/usr/bin/env python3
"""
Database migration script to create personalization tables
This script creates the tables for user personalization preferences and bonus points tracking
"""

import sys
import os

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from database import engine
from models import UserPersonalizationPreference, UserBonusPoints, Chapter

def create_personalization_tables():
    """Create the personalization-related tables in the database"""
    print("Creating personalization tables...")

    # Create all tables defined in models
    UserPersonalizationPreference.metadata.create_all(engine)
    UserBonusPoints.metadata.create_all(engine)
    Chapter.metadata.create_all(engine)

    print("Personalization tables created successfully!")
    print("- user_personalization_preferences")
    print("- user_bonus_points")
    print("- chapters")

if __name__ == "__main__":
    create_personalization_tables()


