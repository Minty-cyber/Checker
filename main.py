import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route.api import router as api_router
from core.config import settings

# from contextlib import asynccontextmanager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
# await initialize_processor()
# yield

app = FastAPI(
    title="Merve API",
    description="API for document processing and RAG-based chat",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    return {"message": "Hello World!"}

# Healthcheck route (needed for Leapcell monitoring)
@app.get("/kaithheathcheck")
async def healthcheck():
    return {"status": "ok"}

# Include API routes
app.include_router(api_router)

# Server binding configuration
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable, default to 8000
    port = settings.PORT
    
    print(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        "main:app",  # Change this to match your filename if different
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True only for development
        log_level="info"
    )