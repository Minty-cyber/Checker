from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route.api import router as api_router
# from contextlib import asynccontextmanager

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await initialize_processor()
#     yield

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
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

# Include API routes
app.include_router(api_router)
