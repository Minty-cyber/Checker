from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field


class DocProcessRequest(BaseModel):
    docs_path: str = Field(..., description="path to the PDF file")
    chunk_size: int = Field(1000, gt=0, description="Size of text chunks")## Larger chunk size for retrieval of summaries
    ## used for maintaining the context of the contents when generating responses. 10-20% of teh chunk_size
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks")
    
    
class ProcessResult(BaseModel):
    status: str
    docs_path: str
    total_chunks: Optional[int] = None
    stored_chunks: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


# class ChunkBase(BaseModel):

class ChunkResponse(BaseModel):
    text: str
    doc_index: int 
    chunk_index: int 
    chunk_id: str
    

    
    