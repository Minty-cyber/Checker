import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from core.config import settings

PINECONE_API_KEY = settings.PINECONE_API_KEY
GROQ_API_KEY = settings.GROQ_API_KEY
INDEX_NAME = "mtn-ethics-index"

# === Pinecone init ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === Embeddings ===
embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY,
    model="embed-english-light-v3.0"
)

# === LLM ===
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

# === Relevance Check Prompt ===
RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict relevance checker for an HR assistant that ONLY answers questions about MTN's Code of Ethics and workplace policies.

Respond with ONLY "RELEVANT" or "NOT_RELEVANT".

RELEVANT questions MUST be about:
- MTN Code of Ethics specifically
- Workplace conduct and behavior policies
- Employee disciplinary procedures
- Business ethics and compliance at work
- Conflict of interest policies
- Gift and entertainment policies in business context
- Harassment, discrimination, or workplace violations
- Employee rights and responsibilities at MTN
- Company compliance and regulatory matters

NOT_RELEVANT questions include:
- Personal life events (weddings, parties, celebrations)
- Personal advice or planning (even if mentioning gifts)
- General knowledge or how-to questions
- Technical support, travel, cooking, entertainment
- Personal finance, shopping, recipes
- Any question about personal matters outside of work

Key rule: If the question is about personal life or personal events (like "help me prepare for a wedding"), it is NOT_RELEVANT even if it mentions business concepts like gifts.

Examples:
- "What gifts can I accept from clients?" → RELEVANT
- "Help me prepare for a wedding" → NOT_RELEVANT  
- "What are MTN's gift policies?" → RELEVANT
- "What gifts should I give at my wedding?" → NOT_RELEVANT"""),
    ("human", "Question: {question}")
])

# === Main Answer Prompt (with context) ===
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are AskHR, an assistant that answers questions strictly using the MTN Code of Ethics.

Rules:
- Answer based strictly on the provided context.
- Cite page numbers from the context in square brackets. Example: [Page 12].
- Keep answers clear and concise.
- Never make up information not in the context."""),
    ("human", "{question}\n\nContext:\n{context}")
])

# === No Context Answer Prompt ===
NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are AskHR, an assistant for MTN employees. 

When no relevant context from the MTN Code of Ethics is available for a question, respond with EXACTLY:
"Kindly ask only questions pertaining to HR."

Do not provide any other response or explanation."""),
    ("human", "{question}")
])

def detect_page(user_query: str) -> int | None:
    """Extract page number if the query refers to a specific page."""
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

def retrieve_chunks(query: str, top_k: int = 8, similarity_threshold: float = 0.35, page: int | None = None):
    query_vector = embeddings.embed_query(query)

    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if page is not None:
        query_params["filter"] = {"page": {"$eq": page}}

    results = index.query(**query_params)

    if page is not None:
        # For explicit page queries, return all matches (ignore threshold)
        return results["matches"]

    # Otherwise, apply similarity filtering
    return [
        m for m in results["matches"]
        if m.get("score", 0) >= similarity_threshold
    ]
   
        
def check_relevance(user_query: str, debug: bool = False) -> bool:
    """Use LLM to check if query is HR-=related."""
    messages = RELEVANCE_PROMPT.format_messages(question=user_query)
    response = llm.invoke(messages)
    relevance_result = response.content.strip().upper()
    
    if debug:
        print(f"🔍 Relevance check for '{user_query}': {relevance_result}")
    
    return relevance_result == "RELEVANT"

def query_rag(user_query: str, top_k: int = 8, similarity_threshold: float = 0.35, debug: bool = False):
    """RAG system with page detection + retrieval (no relevance prompt)."""
    
    if debug:
        print(f"🔍 Processing query: '{user_query}'")
    
    # 1. Detect page (optional)
    requested_page = detect_page(user_query)
    if debug and requested_page:
        print(f"🔍 Detected page request: {requested_page}")
    
    # 2. Retrieve chunks
    relevant_chunks = retrieve_chunks(
        user_query, 
        top_k=top_k, 
        similarity_threshold=similarity_threshold, 
        page=requested_page
    )
    
    if debug:
        print(f"🔍 Found {len(relevant_chunks)} chunks above threshold {similarity_threshold}")
        if relevant_chunks:
            print(f"🔍 Top similarity score: {relevant_chunks[0].get('score', 0):.4f}")
    
    # 3. Handle no results
    if not relevant_chunks:
        if requested_page is not None:
            return {
                "question": user_query,
                "answer": f"No content was found for page {requested_page} of the MTN Code of Ethics.",
                "sources": []
            }
        else:
            return {
                "question": user_query,
                "answer": "Kindly ask questions pertaining to only HR",
                "sources": []
            }
    
    # 4. Build context
    context = ""
    supporting_chunks = []
    for m in relevant_chunks:
        page = m["metadata"].get("page")
        text = m["metadata"].get("text")
        context += f"\n[Page {page}] {text}\n"
        supporting_chunks.append({
            "page": page,
            "text": text[:300] + "..." if len(text) > 300 else text,
            "score": m.get("score")
        })
    
    # 5. Answer with context
    messages = ANSWER_PROMPT.format_messages(
        question=user_query,
        context=context
    )
    response = llm.invoke(messages)
    
    return {
        "question": user_query,
        "answer": response.content,
        "sources": supporting_chunks
    }

if __name__ == "__main__":
    queries = [
        "What is on page 11?",
        "Help me plan for a wedding",
        "What is the best recipe for pizza?",
        "What does the code say about conflict of interest?", 
        "What is the MTN code of Ethics about?",
        "How do I book a flight?",
        "What are the gift policies at MTN?",
        "Who is the president of Ghana?",
        "What is the capital of France?",
        "How do I cook rice?",
        "Help me with my personal finances"
    ]
    
    for q in queries:
        result = query_rag(q, debug=True)  # Enable debug mode
        print(f"\n🔎 Question: {result['question']}")
        print(f"💡 Answer:\n{result['answer']}\n")
        if result['sources']:
            print("📚 Supporting Sources:")
            for src in result['sources'][:3]:
                print(f"- Page {src['page']} (Score {src['score']:.4f}): {src['text']}")
        print("=" * 100)