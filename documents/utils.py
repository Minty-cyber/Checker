import uuid
import asyncio
import os
import hashlib
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import spacy
from llama_parse import LlamaParse
from core.config import settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

from .schemas import DocProcessRequest, ProcessResult, ChunkResponse


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Load SpaCy for NER
nlp = spacy.load("en_core_web_sm")


class DocProcessor:
    def __init__(self):
        self.llamaparse = LlamaParse(
            api_key=settings.LLAMA_API_KEY,
            result_type="text",
            verbose=True,
            language="en",
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"  # Local persistent storage
        )
        
        
        self.collection_name = "doc_chunks"
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings manually
            )
            print(f"Collection '{self.collection_name}' already exists")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # We'll provide embeddings manually
                metadata={"description": "Document chunks with metadata"}
            )
            print(f"Created collection '{self.collection_name}'")

    async def process_docs(self, data_in: DocProcessRequest) -> ProcessResult:
        try:
            start_time = time.time()
            print(f"Started processing: {data_in.docs_path}")

            print("Parsing documents...")
            documents = await self._parse_docs(data_in.docs_path)

            print("Creating chunks...")
            chunks = self._create_chunks(
                documents=documents,
                chunk_size=data_in.chunk_size,
                chunk_overlap=data_in.chunk_overlap,
            )

            print("Storing chunks...")
            stored_chunks = await self._store_chunks(
                chunks=chunks, docs_path=data_in.docs_path
            )

            elapsed_time = time.time() - start_time
            return ProcessResult(
                status="success",
                docs_path=data_in.docs_path,
                total_chunks=len(chunks),
                stored_chunks=stored_chunks,
                message=f"Successfully processed {Path(data_in.docs_path).name} in {elapsed_time:.2f} seconds",
            )
        except Exception as e:
            return ProcessResult(
                status="error", docs_path=data_in.docs_path, error=str(e)
            )

    async def _parse_docs(self, docs: str) -> List[str]:
        try:
            documents = await self.llamaparse.aload_data(docs)
            return [doc.text for doc in documents]
        except Exception:
            return []

    def _create_chunks(self, documents: List[str], chunk_size: int, chunk_overlap: int) -> List[ChunkResponse]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        chunks = []
        raw_word_count = 0
        chunked_word_count = 0

        for doc_idx, document in enumerate(documents):
            raw_word_count += len(document.split())
            split_texts = splitter.split_text(document)

            for idx, text in enumerate(split_texts):
                chunks.append(
                    ChunkResponse(
                        text=text.strip(),
                        doc_index=doc_idx,
                        chunk_index=idx,
                        chunk_id=f"doc_{doc_idx}_chunk_{idx}",
                    )
                )
                chunked_word_count += len(text.split())

        print("\n------ Document Integrity Check ------")
        print(f"✅ Raw words: {raw_word_count}")
        print(f"✅ Chunked words: {chunked_word_count}")
        print(f"✅ Total chunks: {len(chunks)}")
        if abs(raw_word_count - chunked_word_count) > raw_word_count * 0.02:
            print("⚠️ Warning: Possible mismatch between raw and chunked content!")
        print("-------------------------------------\n")

        return chunks

    def _extract_metadata(self, 
        chunk_text: str, 
        docs_path: str, 
        i: int, 
        chunk, 
        file_hash: str, 
        true_page_number: int
    ):
        """Extract metadata with defaults if missing"""
        doc = nlp(chunk_text)

        # Detect contacts (emails)
        email_match = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", chunk_text)
        contact_info = email_match[0] if email_match else None

        # Named entities
        entities = [ent.text for ent in doc.ents]
        print(entities)

        # Keywords (simple nouns + proper nouns)
        keywords = [token.text for token in doc if token.is_alpha and token.pos_ in ["NOUN", "PROPN"]]

        # Rule type classification
        lower_text = chunk_text.lower()
        if "must not" in lower_text or "prohibited" in lower_text:
            rule_type = "prohibition"
        elif "must" in lower_text or "required" in lower_text:
            rule_type = "obligation"
        elif "should" in lower_text or "encouraged" in lower_text:
            rule_type = "guideline"
        else:
            rule_type = "other"

        return {
            "document_id": str(uuid.uuid4()),
            "document_title": "MTN Code of Ethics",
            "company": "Scancom Ltd. (MTN Ghana)",
            "publication_date": None,
            "version": "N/A",
            "source_file": Path(docs_path).name,
            "file_hash": file_hash,
            "file_type": "pdf",

            "page_number": true_page_number,
            "page_position": "middle",

            "section_heading": "unknown",
            "section_hierarchy": [],
            "paragraph_index": chunk.chunk_index,
            "content_type": "paragraph",

            "keywords": keywords,
            "entities": entities,
            "contains_numbers": any(ch.isdigit() for ch in chunk_text),
            "contains_contacts": bool(contact_info),

            "rule_type": rule_type,
            "applies_to": ["employees"],  # default
            "sanction": None,

            "references": [],
            "contact_info": contact_info,

            "chunk_id": chunk.chunk_id,
        }

    async def _store_chunks(self, chunks: List[ChunkResponse], docs_path: str, batch_size: int = 100) -> int:
        total_chunks = len(chunks)
        stored_count = 0
        print(f"Processing {total_chunks} chunks...")

        file_hash = self._get_file_hash(docs_path)

        # Prepare batch data for Chroma
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []
        batch_documents = []

        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk.text)
            point_id = str(uuid.uuid4())

            metadata = self._extract_metadata(
                chunk.text, docs_path, i, chunk, file_hash, 
                true_page_number=chunk.doc_index+1
            )
            
            # Convert lists to strings for Chroma metadata (Chroma doesn't support nested structures)
            metadata_flat = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata_flat[key] = json.dumps(value) if value else "[]"
                elif value is None:
                    metadata_flat[key] = ""
                else:
                    metadata_flat[key] = str(value)

            batch_ids.append(point_id)
            batch_embeddings.append(embedding.tolist())
            batch_metadatas.append(metadata_flat)
            batch_documents.append(chunk.text)

            # Store in batches
            if len(batch_ids) >= batch_size:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                stored_count += len(batch_ids)
                
                # Reset batches
                batch_ids = []
                batch_embeddings = []
                batch_metadatas = []
                batch_documents = []

        # Store remaining chunks
        if batch_ids:
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
            stored_count += len(batch_ids)

        return stored_count

    def _get_file_hash(self, docs_path: str) -> str:
        hash_sha256 = hashlib.sha256()
        with open(docs_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    # ----------------- Retrieval Functions -----------------

    def search_by_page(self, page: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch all chunks from a specific page with metadata."""
        try:
            results = self.collection.get(
                where={"page_number": str(page)},
                limit=limit,
                include=["metadatas", "documents"]
            )
            
            formatted_results = []
            for i, (metadata, document) in enumerate(zip(results["metadatas"], results["documents"])):
                # Parse back JSON strings to lists where needed
                parsed_metadata = {}
                for key, value in metadata.items():
                    if key in ["keywords", "entities", "section_hierarchy", "references", "applies_to"]:
                        try:
                            parsed_metadata[key] = json.loads(value) if value else []
                        except:
                            parsed_metadata[key] = []
                    else:
                        parsed_metadata[key] = value
                
                parsed_metadata["text"] = document
                formatted_results.append(parsed_metadata)
            
            return formatted_results
        except Exception as e:
            print(f"Error in search_by_page: {e}")
            return []

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Keyword search using Chroma's query functionality."""
        try:
            results = self.collection.query(
                query_texts=[keyword],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            formatted_results = []
            if results["metadatas"]:
                for metadata, document in zip(results["metadatas"][0], results["documents"][0]):
                    # Parse back JSON strings to lists where needed
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        if key in ["keywords", "entities", "section_hierarchy", "references", "applies_to"]:
                            try:
                                parsed_metadata[key] = json.loads(value) if value else []
                            except:
                                parsed_metadata[key] = []
                        else:
                            parsed_metadata[key] = value
                    
                    parsed_metadata["text"] = document
                    formatted_results.append(parsed_metadata)
            
            return formatted_results
        except Exception as e:
            print(f"Error in search_by_keyword: {e}")
            return []

    def _make_key(self, query: str, filters: Dict = None, limit: int = None) -> str:
        components = [query.strip().lower()]  # Normalize query
        
        if filters:
            # Sort filter items for consistent keys and handle different data types
            filter_items = []
            for k, v in sorted(filters.items()):
                # Handle different value types consistently
                if isinstance(v, (list, tuple)):
                    v_str = ",".join(str(item) for item in sorted(v))
                else:
                    v_str = str(v)
                filter_items.append(f"{k}:{v_str}")
            components.append("_".join(filter_items))
        
        if limit:
            components.append(str(limit))
            
        key_string = "|".join(components)
        key_hash = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
        cache_key = f"search:{key_hash}"
        
        # Debug logging
        print(f"🔑 Cache key components: {components}")
        print(f"🔑 Final key string: {key_string}")
        print(f"🔑 Cache key: {cache_key}")
        
        return cache_key

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            # Manual embedding
            query_embedding = self.embedding_model.encode([keyword]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )

            formatted_results = []
            if results["metadatas"]:
                for metadata, document in zip(results["metadatas"][0], results["documents"][0]):
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        if key in ["keywords", "entities", "section_hierarchy", "references", "applies_to"]:
                            try:
                                parsed_metadata[key] = json.loads(value) if value else []
                            except:
                                parsed_metadata[key] = []
                        else:
                            parsed_metadata[key] = value

                    parsed_metadata["text"] = document
                    formatted_results.append(parsed_metadata)

            # Extra keyword filter (regex match in text)
            keyword_matches = [
                doc for doc in formatted_results
                if re.search(keyword, doc["text"], re.IGNORECASE)
            ]

            return keyword_matches[:limit] if keyword_matches else formatted_results
        except Exception as e:
            print(f"Error in search_by_keyword: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:

        query = query.strip()
        filters = filters or {}


        # Manual embedding for semantic search
        query_embedding = self.embedding_model.encode([query]).tolist()

        where_clause = {}
        if filters:
            for key, value in filters.items():
                where_clause[key] = str(value)

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit * 2,
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"]
            )

            candidates = []
            if results["metadatas"]:
                for metadata, document in zip(results["metadatas"][0], results["documents"][0]):
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        if key in ["keywords", "entities", "section_hierarchy", "references", "applies_to"]:
                            try:
                                parsed_metadata[key] = json.loads(value) if value else []
                            except:
                                parsed_metadata[key] = []
                        else:
                            parsed_metadata[key] = value

                    parsed_metadata["text"] = document
                    candidates.append(parsed_metadata)

        except Exception as e:
            print(f"Error in semantic search: {e}")
            candidates = []

        if not candidates:
            print("⚠️ No strong semantic match. Falling back to keyword search...")
            candidates = self.search_by_keyword(query, limit=limit * 2)

        if candidates:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )

            final_results = [
                {**c, "rerank_score": float(s)} for c, s in reranked[:limit]
            ]


            return final_results

        return []

# ... [all your imports and class DocProcessor remain unchanged] ...


processor = DocProcessor()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ----------------- CLI Test -----------------
async def main():
    docs_path = BASE_DIR / "documents/mtn_code_of_ethics.pdf"

    if os.path.exists(docs_path):
        request = DocProcessRequest(
            docs_path=docs_path, chunk_size=1000, chunk_overlap=200
        )
        result = await processor.process_docs(request)
        print(f"Processing result: {result}")

        if result.status == "success":
            res1 = processor.search_by_page(3)
            print(f"Page 3 results: {res1}")

            res2 = processor.search_hybrid("Gift Policy clause 12.1.5")
            print(f"Hybrid search results: {res2}")
    else:
        print(f"PDF file not found: {docs_path}")


if __name__ == "__main__":
    asyncio.run(main())
