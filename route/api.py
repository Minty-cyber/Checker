# app/main.py
import time
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_groq import ChatGroq
from core.config import settings

# === Config ===
PINECONE_API_KEY = settings.PINECONE_API_KEY
GROQ_API_KEY = settings.GROQ_API_KEY
INDEX_NAME = "mtn-ethics-index"

# === Init ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY,  # or set COHERE_API_KEY env var
    model="embed-english-light-v3.0"  # or "embed-multilingual-v3.0"
)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

SYSTEM_PROMPT = """
You are a document assistant called MERVE. You can answer user pleasantries and small greetings.
Always answer based strictly on the provided context in your knowledge base (MTN Code of Ethics).

- If the user mentions a page or range of pages, use the `metadata.page` values of the retrieved chunks.
- If no page is mentioned, answer using the most relevant chunks.
- Always provide clear, concise answers.
- Never make up information. If unsure, say "I could not find that in the document."
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}\n\nContext:\n{context}")
])

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10

class QueryResponse(BaseModel):
    question: str
    answer: str
    latency: float
    sources: list


# --- Page detection using LLM ---
def detect_page(user_query: str) -> int | None:
    """Detect if a query refers to a specific page number using the LLM."""
    page_prompt = f"""
    Extract the page number if the query refers to a specific page.
    Query: "{user_query}"
    Answer with ONLY the page number (integer). If no page is mentioned, answer 'None'.
    """
    response = llm.invoke([{"role": "user", "content": page_prompt}])
    try:
        content = response.content.strip()
        if content.lower() == "none":
            return None
        return int(content)
    except:
        return None


# --- Pinecone retrieval with optional page filter ---
def retrieve_chunks(query: str, top_k: int = 10, page: int | None = None):
    query_vector = embeddings.embed_query(query)

    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if page is not None:
        query_params["filter"] = {"page": {"$eq": page}}

    results = index.query(**query_params)
    return results["matches"]


# --- Main RAG pipeline ---
def query_rag(user_query: str, top_k: int = 10):
    start = time.time()

    page = detect_page(user_query)
    matches = retrieve_chunks(user_query, top_k=top_k, page=page)

    context = ""
    supporting_chunks = []
    for m in matches:
        page_num = m["metadata"].get("page")
        text = m["metadata"].get("text")
        context += f"\n[Page {page_num}] {text}\n"
        supporting_chunks.append({
            "page": page_num,
            "text": text[:300] + "..." if len(text) > 300 else text,
            "score": m.get("score")
        })

    messages = prompt_template.format_messages(
        question=user_query,
        context=context
    )
    response = llm.invoke(messages)

    latency = round(time.time() - start, 2)

    return {
        "question": user_query,
        "answer": response.content,
        "latency": latency,
        "sources": supporting_chunks
    }


@router.post("/query", response_model=QueryResponse)
async def query_api(request: QueryRequest):
    return query_rag(request.question, top_k=request.top_k)
