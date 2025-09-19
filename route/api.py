from fastapi import APIRouter, HTTPException, Depends, FastAPI
from typing import List, Optional
from functools import lru_cache
from pathlib import Path
import os
from datetime import datetime

from documents.schemas import DocProcessRequest, ProcessResult
from documents.utils import DocProcessor
from agent import agent_responder
from pydantic import BaseModel, Field

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent  # go up one level from "route"
DOCS_PATH = BASE_DIR / "documents/mtn_code_of_ethics.pdf"

router = APIRouter()

# Global processor instance
_processor: DocProcessor | None = None


@lru_cache
def get_processor() -> DocProcessor:
    """Get or create the DocProcessor instance"""
    global _processor
    if _processor is None:
        raise RuntimeError("DocProcessor not initialized")
    return _processor


async def initialize_processor():
    """Initialize the global processor instance and process PDF if needed"""
    global _processor
    if _processor is None:
        _processor = DocProcessor()
        print("✅ Document Processor initialized successfully")

        try:
            # --- Check collection size ---
            collection_data = _processor.collection.get()
            count = len(collection_data["ids"])
            if count == 0:
                print("⚠️ Collection is empty — processing documents...")
                if DOCS_PATH.exists():
                    request = DocProcessRequest(
                        docs_path=str(DOCS_PATH),
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    result = await _processor.process_docs(request)
                    print(f"✅ Processing result: {result}")
                else:
                    print(f"❌ PDF file not found: {DOCS_PATH}")
            else:
                print(f"✅ Collection already has {count} docs — skipping processing")
        except Exception as e:
            print(f"❌ Failed to check or process collection: {e}")


# ---- Pydantic Models ----
class AgentQuery(BaseModel):
    query: str = Field(..., description="The question to ask the agent")
    thread_id: Optional[str] = Field(
        "api-thread", description="Thread ID for conversation continuity"
    )


class Message(BaseModel):
    content: str


class AgentResponse(BaseModel):
    status: str = Field("success")
    message: str = Field("", description="The response message")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used for the response"
    )
    query: str = Field(..., description="Original query from the user")
    timestamp: datetime = Field(default_factory=datetime.now)
    route: str = Field("unknown")
    messages: List[Message]


# ---- FastAPI App ----
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    await initialize_processor()


@router.post("/chat", response_model=AgentResponse)
async def agent_chat(query_data: AgentQuery):
    try:
        # Unpack query
        query = query_data.query
        print(query)

        # Build config
        config = {"configurable": {"thread_id": query_data.thread_id}}
        result = agent_responder(query, config)

        # Extract latest message
        latest_message = result["messages"][-1] if result.get("messages") else None
        message_content = latest_message["content"] if latest_message else ""
        result["message"] = message_content

        # Ensure sources only included for rag
        if result.get("route") != "rag":
            result["sources"] = []

        # Add timestamp
        result["timestamp"] = datetime.now()

        # Validate and return using Pydantic model
        return AgentResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e),
                "sources": [],
                "query": query_data.query,
                "timestamp": datetime.now(),
                "route": "error",
                "messages": [],
            },
        )


@app.get("/agent/health")
async def agent_health():
    return {
        "status": "healthy",
        "service": "LangGraph Agent API",
        "timestamp": datetime.now(),
    }
