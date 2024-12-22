import logging
import os
from typing import Dict, Type

from preprocess.loader.base_loader import DocumentLoader
from preprocess.loader.csv_loader import CSVDocLoader
from preprocess.loader.docx_loader import DocxDocLoader
from preprocess.loader.json_loader import JsonDocLoader
from preprocess.loader.pdf_loader import PDFDocLoader
from preprocess.loader.text_loader import TextDocLoader

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    loader_mapping : Dict[str, Type[DocumentLoader]] = {
        'pdf': PDFDocLoader,
        'txt': TextDocLoader,
        'csv': CSVDocLoader,
        'json': JsonDocLoader,
        'docx': DocxDocLoader,
    }

    # Factory method
    @staticmethod
    def get_loader(file_extension: str) -> DocumentLoader:
        if not file_extension:
            raise ValueError("Invalid file extension")

        # check if the file_extension is in the supported list - key in loader_mapping
        if file_extension not in DocumentLoaderFactory.loader_mapping:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        loader_type = DocumentLoaderFactory.loader_mapping.get(file_extension, TextDocLoader)
        return loader_type()
