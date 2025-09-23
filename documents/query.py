import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # or OpenAI if you prefer
from core.config import settings



PINECONE_API_KEY = settings.PINECONE_API_KEY
GROQ_API_KEY = settings.GROQ_API_KEY
INDEX_NAME = "mtn-ethics-index"

# === Pinecone init ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === Embeddings ===
embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY,  # or set COHERE_API_KEY env var
    model="embed-english-light-v3.0"  # or "embed-multilingual-v3.0"
)

# === LLM (Groq here, but you can swap with OpenAI) ===
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

# === Prompt ===
SYSTEM_PROMPT = """
You are AskHR, an assistant that answers questions strictly using the MTN Code of Ethics.

Rules:
- Always answer based strictly on the provided context.
- If the user mentions a page or range of pages, cite the page numbers from the context in square brackets. Example: [Page 12].
- If no relevant context exists for the question (unrelated to HR or not found in the Code of Ethics), reply strictly:
  "Kindly ask only questions pertaining to HR."
- Do not hallucinate, do not make up sources.
- Only include citations (page numbers) when the context is relevant and used in your answer.
- Never prepend answers with phrases like "based on the provided context".
- Keep answers clear and concise.
"""


prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}\n\nContext:\n{context}")
])

def retrieve_chunks(query: str, top_k: int = 8):
    """Retrieve top_k relevant chunks from Pinecone."""
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

def query_rag(user_query: str, top_k: int = 8):
    # 1. Retrieve supporting chunks
    matches = retrieve_chunks(user_query, top_k=top_k)
    
    # 2. Build context
    context = ""
    supporting_chunks = []
    for m in matches:
        page = m["metadata"].get("page")
        text = m["metadata"].get("text")
        context += f"\n[Page {page}] {text}\n"
        supporting_chunks.append({
            "page": page,
            "text": text[:300] + "..." if len(text) > 300 else text,
            "score": m.get("score")
        })
    
    # 3. Build prompt
    messages = prompt_template.format_messages(
        question=user_query,
        context=context
    )
    
    # 4. Get final answer from LLM
    response = llm.invoke(messages)
    
    # 5. Return both answer + citations
    return {
        "question": user_query,
        "answer": response.content,
        # "sources": supporting_chunks
    }

if __name__ == "__main__":
    queries = [
        # "What is on page 2?",
        # "Summarise page 4 for me",
        # "Summarise page 1 to 3",
        # "What is the MTN code of Ethics about?",
        # "What does the code say about conflict of interest?",
        "Is there any email in the document in your knowledge base?"
    ]
    
    for q in queries:
        result = query_rag(q)
        print(f"\n🔎 Question: {result['question']}")
        print(f"💡 Answer:\n{result['answer']}\n")
        print("📚 Supporting Sources:")
        for src in result["sources"][:3]:  # show first 3
            print(f"- Page {src['page']} (Score {src['score']:.4f}): {src['text']}")
        print("="*100)
