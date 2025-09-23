# import uuid
# import asyncio
# import os
# import hashlib
# import time
# from pathlib import Path
# from typing import List, Dict, Any

# from llama_parse import LlamaParse
# from core.config import settings
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec
# from .schemas import DocProcessRequest, ProcessResult, ChunkResponse


# class DocProcessor:
#     def __init__(self):
#         # Parser
#         self.llamaparse = LlamaParse(
#             api_key=settings.LLAMA_API_KEY,
#             result_type="text",
#             verbose=True,
#             language="en",
#         )

#         # Embeddings + reranker
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#         # Pinecone
#         self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
#         self.index_name = "doc-chunks"
#         self._create_index()
#         self.index = self.pc.Index(self.index_name)

#     def _create_index(self):
#         if self.index_name not in [i["name"] for i in self.pc.list_indexes()]:
#             self.pc.create_index(
#                 name=self.index_name,
#                 dimension=384,  # embedding dim
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#             )

#     async def process_docs(self, data_in: DocProcessRequest) -> ProcessResult:
#         try:
#             start_time = time.time()
#             print(f"Started processing: {data_in.docs_path}")

#             # Parse
#             documents = await self._parse_docs(data_in.docs_path)
#             print(f"Parsed {len(documents)} docs")

#             # Chunk
#             chunks = self._create_chunks(
#                 documents, data_in.chunk_size, data_in.chunk_overlap
#             )
#             print(f"Created {len(chunks)} chunks")

#             # Store
#             stored_chunks = await self._store_chunks(chunks, data_in.docs_path)
#             elapsed = time.time() - start_time

#             return ProcessResult(
#                 status="success",
#                 docs_path=data_in.docs_path,
#                 total_chunks=len(chunks),
#                 stored_chunks=stored_chunks,  # ✅ normalized list of chunks
#                 message=f"✅ Processed {Path(data_in.docs_path).name} in {elapsed:.2f}s",
#             )
#         except Exception as e:
#             return ProcessResult(
#                 status="error", docs_path=data_in.docs_path, error=str(e)
#             )

#     async def _parse_docs(self, docs_path: str) -> List[str]:
#         docs = await self.llamaparse.aload_data(docs_path)
#         return [doc.text for doc in docs]

#     def _create_chunks(
#         self, documents: List[str], chunk_size: int, chunk_overlap: int
#     ) -> List[ChunkResponse]:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", ". ", " "],
#         )
#         chunks = []
#         for doc_idx, doc in enumerate(documents):
#             for idx, text in enumerate(splitter.split_text(doc)):
#                 chunks.append(
#                     ChunkResponse(
#                         text=text.strip(),
#                         doc_index=doc_idx,
#                         chunk_index=idx,
#                         chunk_id=f"doc{doc_idx}_chunk{idx}",
#                     )
#                 )
#         return chunks

#     async def _store_chunks(
#         self, chunks: List[ChunkResponse], docs_path: str, batch_size: int = 100
#     ) -> List[Dict[str, Any]]:
#         vectors, stored = [], []
#         file_hash = self._get_file_hash(docs_path)

#         for chunk in chunks:
#             embedding = self.embedding_model.encode(chunk.text).tolist()
#             metadata = {
#                 "chunk_id": chunk.chunk_id,
#                 "page_number": chunk.doc_index + 1,
#                 "chunk_index": chunk.chunk_index,
#                 "source_file": Path(docs_path).name,
#                 "chunk_text": chunk.text[:1000],
#                 "file_hash": file_hash,
#             }

#             vec = {"id": str(uuid.uuid4()), "values": embedding, "metadata": metadata}
#             vectors.append(vec)

#             # ✅ normalize here
#             stored.append(
#                 {
#                     "content": metadata["chunk_text"],
#                     "metadata": {k: v for k, v in metadata.items() if k != "chunk_text"},
#                 }
#             )

#             if len(vectors) >= batch_size:
#                 self.index.upsert(vectors=vectors)
#                 vectors = []

#         if vectors:
#             self.index.upsert(vectors=vectors)

#         return stored

#     def _get_file_hash(self, docs_path: str) -> str:
#         h = hashlib.sha256()
#         with open(docs_path, "rb") as f:
#             for chunk in iter(lambda: f.read(4096), b""):
#                 h.update(chunk)
#         return h.hexdigest()

#     # ---------------- Retrieval ----------------

#     def search_by_page(self, page: int, limit: int = 5):
#         results = self.index.query(
#             vector=[0.0] * 384,  # dummy vector
#             top_k=limit,
#             filter={"page_number": page},
#             include_metadata=True,
#         )
#         normalized = []
#         for m in results["matches"]:
#             meta = m["metadata"]
#             normalized.append(
#                 {
#                     "content": meta.get("chunk_text", ""),
#                     "metadata": {k: v for k, v in meta.items() if k != "chunk_text"},
#                 }
#             )
#         return normalized

#     def search_hybrid(self, query: str, limit: int = 5):
#         query_emb = self.embedding_model.encode(query).tolist()
#         results = self.index.query(
#             vector=query_emb, top_k=limit * 2, include_metadata=True
#         )
#         candidates = []
#         for m in results["matches"]:
#             meta = m["metadata"]
#             candidates.append(
#                 {
#                     "content": meta.get("chunk_text", ""),
#                     "metadata": {k: v for k, v in meta.items() if k != "chunk_text"},
#                 }
#             )

#         if not candidates:
#             return []

#         # rerank
#         pairs = [(query, c["content"]) for c in candidates]
#         scores = self.reranker.predict(pairs)
#         reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

#         return [{**c, "rerank_score": float(s)} for c, s in reranked[:limit]]


# # ----------------- Setup -----------------
# processor = DocProcessor()
# BASE_DIR = Path(__file__).resolve().parent.parent


# # ----------------- CLI Test -----------------
# # ----------------- CLI Test -----------------
# async def main():
#     docs_path = BASE_DIR / "documents/mtn_code_of_ethics.pdf"

#     if os.path.exists(docs_path):
#         request = DocProcessRequest(
#             docs_path=str(docs_path), chunk_size=1000, chunk_overlap=200
#         )

#         # Process and store chunks
#         result = await processor.process_docs(request)
#         print(f"Processing result: {result.message}")

#         if result.status == "success":
#             print(f"\nStored {len(result.stored_chunks)} chunks")

#             # 🔍 Debug retrieval with your query
#             query = "What is the MTN code of Ethics about?"
#             print(f"\n=== Hybrid search for query: '{query}' ===")
#             res_debug = processor.search_hybrid(query, limit=5)
#             for r in res_debug:
#                 print(
#                     f"Score: {r.get('rerank_score', 0):.4f}\n"
#                     f"Content: {r['content']}\n"
#                     f"Metadata: {r['metadata']}\n"
#                 )

#             # Also check a page retrieval to ensure chunks are tied to pages
#             res_page = processor.search_by_page(1)
#             print("\n=== Page 1 results ===")
#             for r in res_page:
#                 print(f"Content: {r['content']}\nMetadata: {r['metadata']}\n")

#     else:
#         print(f"❌ PDF file not found: {docs_path}")


# if __name__ == "__main__":
#     asyncio.run(main())
