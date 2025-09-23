from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "documents" / "mtn_code_of_ethics.pdf"

def extract_pdf_with_metadata(pdf_path: Path, chunk_size=1000, chunk_overlap=200):
    """
    Extracts page-aware chunks from a PDF using PyMuPDF.
    Each chunk has page number, unique chunk_id, and source filename.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    print(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    global_chunk_id = 0  # ✅ ensures unique IDs across whole PDF
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue
        
        # Split long page text into smaller chunks
        parts = splitter.split_text(text)
        for part in parts:
            chunks.append({
                "page": page_num + 1,   # 1-indexed
                "chunk_id": global_chunk_id,
                "text": part,
                "source": pdf_path.name
            })
            global_chunk_id += 1  # ✅ increment globally
    
    doc.close()
    return chunks

if __name__ == "__main__":
    chunks = extract_pdf_with_metadata(DOCS_PATH)
    print(f"✅ Extracted {len(chunks)} chunks from {DOCS_PATH.name}")
    for c in chunks:
        print(c)
