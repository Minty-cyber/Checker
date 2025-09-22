from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route.api import router as api_router
from contextlib import asynccontextmanager

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

@app.get("/")
async def root():
    return {"message": "Hello World!"}

app.include_router(api_router)
