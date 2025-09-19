# core/processor.py
from typing import Optional
from documents.utils import DocProcessor


_processor_instance: Optional[DocProcessor] = None

def get_processor() -> DocProcessor:
    global _processor_instance
    if _processor_instance is None:
        print("⚡ Initializing DocProcessor...")
        _processor_instance = DocProcessor()
    return _processor_instance
