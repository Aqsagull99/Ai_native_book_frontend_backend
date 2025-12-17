from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from auth import router as auth_router
from endpoints.personalization import router as personalization_router, bonus_points_router
from src.rag_agent.api_service import router as rag_agent_router
import os
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
    allow_origins=["*"],  # In production, replace with specific origins
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
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
