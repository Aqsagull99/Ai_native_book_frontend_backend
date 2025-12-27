from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db, DATABASE_URL, engine
from models import User, UserProfile  # Import models to register them with Base
from auth import router as auth_router
from endpoints.personalization import router as personalization_router, bonus_points_router
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Better-Auth API",
    description="API for user authentication and profile management",
    version="1.0.0"
)

# Import RAG agent router with error handling for missing environment variables
try:
    from src.rag_agent.api_service import router as rag_agent_router
    rag_agent_available = True
except Exception as e:
    print(f"Warning: RAG agent could not be initialized: {str(e)}")
    print("RAG functionality will be disabled")
    # Create a dummy router that returns error messages
    from fastapi import APIRouter, HTTPException
    rag_error_router = APIRouter(prefix="/api/rag", tags=["rag-agent"])

    @rag_error_router.post("/query")
    async def rag_query_error():
        raise HTTPException(status_code=503, detail="RAG service not available - check environment variables")

    @rag_error_router.get("/health")
    async def rag_health_error():
        return {"status": "unavailable", "error": "RAG service not available - check environment variables"}

    rag_agent_router = rag_error_router
    rag_agent_available = False

# Import Translation router with error handling for missing environment variables
try:
    from src.endpoints.translation import router as translation_router
    translation_available = True
except Exception as e:
    print(f"Warning: Translation service could not be initialized: {str(e)}")
    print("Translation functionality will be disabled")
    # Create a dummy router that returns error messages
    from fastapi import APIRouter
    translation_error_router = APIRouter(prefix="/api/translation", tags=["translation"])

    @translation_error_router.post("/translate")
    async def translate_error():
        raise HTTPException(status_code=503, detail="Translation service not available - check OPENROUTER_API_KEY environment variable")

    @translation_error_router.post("/translate-async")
    async def translate_async_error():
        raise HTTPException(status_code=503, detail="Translation service not available - check OPENROUTER_API_KEY environment variable")

    @translation_error_router.post("/status")
    async def status_error():
        raise HTTPException(status_code=503, detail="Translation service not available - check OPENROUTER_API_KEY environment variable")

    @translation_error_router.post("/cache/check")
    async def cache_check_error():
        raise HTTPException(status_code=503, detail="Translation service not available - check OPENROUTER_API_KEY environment variable")

    @translation_error_router.get("/health")
    async def translation_health_error():
        return {"status": "unavailable", "service": "translation", "error": "Translation service not available - check OPENROUTER_API_KEY environment variable"}

    translation_router = translation_error_router
    translation_available = False

# Initialize database tables on startup with timeout handling
@app.on_event("startup")
async def startup_event():
    from models import User, UserProfile  # Import models to ensure they're registered

    print(f"Initializing database tables...")
    print(f"Database URL: {DATABASE_URL}")

    try:
        # For SQLite, we need to use the sync version to create tables
        if DATABASE_URL and DATABASE_URL.startswith("sqlite+aiosqlite"):
            sync_db_url = DATABASE_URL.replace("sqlite+aiosqlite:///", "sqlite:///")

            # Create sync engine for table creation
            sync_engine = create_engine(sync_db_url)

            # Create all tables
            Base.metadata.create_all(bind=sync_engine)
            print("SQLite database tables created successfully!")
        else:
            # For other databases, create tables directly with timeout
            import asyncio
            async with engine.begin() as conn:
                await asyncio.wait_for(conn.run_sync(Base.metadata.create_all), timeout=30.0)
            print("Database tables created successfully!")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Don't fail startup completely if database init fails - let the app start and handle DB errors per request

    # Pre-initialize the RAG agent to avoid long per-request initialization delays
    try:
        # Import and create the agent (api_service handles errors internally)
        from src.rag_agent.api_service import create_rag_agent
        # Use a timeout for RAG agent initialization to prevent startup hangs
        import asyncio
        await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, create_rag_agent), timeout=15.0)
        print("RAG agent pre-initialized at startup")
    except asyncio.TimeoutError:
        print("RAG agent pre-initialization timed out after 15 seconds - will initialize on first request")
    except Exception as e:
        print(f"RAG agent pre-initialization failed: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-native-book-frontend.vercel.app",  # Production frontend
        "https://ai-native-book-frontend-backend.onrender.com",  # Production backend
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Local development alternative
        "http://localhost:5000",  # Docusaurus default
        "http://localhost:3002",  # Docusaurus alternative
        "http://127.0.0.1:3000",  # Local development
        "http://127.0.0.1:3001",  # Local development alternative
        "http://127.0.0.1:5000",  # Docusaurus default
        "http://127.0.0.1:3002",  # Docusaurus alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

# Include personalization routes
app.include_router(personalization_router, prefix="/api", tags=["personalization"])

# Include user bonus points routes
app.include_router(bonus_points_router, prefix="/api", tags=["user"])

# Include RAG agent routes
app.include_router(rag_agent_router, prefix="/api/rag", tags=["rag-agent"])

# Include translation routes
app.include_router(translation_router, tags=["translation"])

@app.get("/")
async def root():
    return {"message": "Better-Auth API is running!"}

@app.get("/health")
async def health_check():
    import os
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "port": os.getenv("PORT", "8000"),
            "database_url_set": bool(os.getenv("DATABASE_URL")),
            "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "qdrant_url_set": bool(os.getenv("QDRANT_URL")),
            "qdrant_api_key_set": bool(os.getenv("QDRANT_API_KEY")),
        },
        "services": {
            "rag_agent_available": rag_agent_available,
            "translation_available": translation_available,
        }
    }
    return health_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))