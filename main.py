from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route.api import router as api_router, initialize_processor
from contextlib import asynccontextmanager
import os
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the document processor at startup
    await initialize_processor()
    yield
    # Cleanup (if needed) at shutdown
    pass


app = FastAPI(
    title="Merve API",
    description="API for document processing and RAG-based chat",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route (like Express `app.get('/')`)
@app.get("/")
async def root():
    return {"message": "Hello World!"}


# Include routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    port = settings.PORT  # same as Express `process.env.PORT || 4000`
    uvicorn.run(app, host="0.0.0.0", port=port)
