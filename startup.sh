#!/bin/bash
# Startup script for Render deployment

set -e  # Exit on any error

echo "Starting application initialization..."

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if required environment variables are set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable is not set"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "ERROR: SECRET_KEY environment variable is not set"
    exit 1
fi

# Run database migrations if needed
echo "Running database initialization..."
python -c "
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Import and run minimal database setup
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, DATABASE_URL

print(f'Database URL: {DATABASE_URL}')

# For SQLite, we need to use the sync version to create tables
if DATABASE_URL and DATABASE_URL.startswith('sqlite+aiosqlite'):
    sync_db_url = DATABASE_URL.replace('sqlite+aiosqlite:///', 'sqlite:///')
    sync_engine = create_engine(sync_db_url)
    Base.metadata.create_all(bind=sync_engine)
    print('SQLite database tables created successfully!')
else:
    # For other databases, create tables directly
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(DATABASE_URL)
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print('Database tables created successfully!')
    asyncio.run(create_tables())
"

echo "Starting uvicorn server..."
# Start the application server
exec uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000} --workers 1