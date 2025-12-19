from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from auth import router as auth_router
from endpoints.personalization import router as personalization_router, bonus_points_router
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Better-Auth API",
    description="API for user authentication and profile management",
    version="1.0.0"
)

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
        }
    }
    return health_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
