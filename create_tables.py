from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from models import User, UserProfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# For SQLite, we need to use the sync version to create tables
if DATABASE_URL and DATABASE_URL.startswith("sqlite+aiosqlite"):
    sync_db_url = DATABASE_URL.replace("sqlite+aiosqlite:///", "sqlite:///")

    # Create sync engine for table creation
    sync_engine = create_engine(sync_db_url, echo=True)

    # Create all tables
    Base.metadata.create_all(bind=sync_engine)

    print("Database tables created successfully!")