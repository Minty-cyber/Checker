# ingest.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_cohere.embeddings import CohereEmbeddings
from .test import extract_pdf_with_metadata, DOCS_PATH
from core.config import settings
INDEX_NAME = "mtn-ethics-index"

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,   # ✅ all-MiniLM-L6-v2 outputs 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ✅ Use SentenceTransformer for embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY,
    model="embed-english-light-v3.0"  # or "embed-multilingual-v3.0"
)

def ingest_pdf():
    chunks = extract_pdf_with_metadata(DOCS_PATH)
    vectors = []
    
    for chunk in chunks:
        vector = embeddings.embed_query(chunk["text"])  # local embedding
        vectors.append({
            "id": f"{chunk['source']}_p{chunk['page']}_c{chunk['chunk_id']}",
            "values": vector,
            "metadata": {
                "page": int(chunk["page"]),
                "source": chunk["source"],
                "text": chunk["text"]
            }
        })
    
    print(f"📤 Upserting {len(vectors)} vectors into Pinecone...")
    index.upsert(vectors)
    print("✅ Done!")

if __name__ == "__main__":
    ingest_pdf()
