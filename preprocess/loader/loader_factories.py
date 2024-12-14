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
    def get_loader(source_type: str) -> DocumentLoader:
        if source_type is None:
            logger.error("Source type is None")
            raise ValueError(f"Source type is Non")

        # if cannot find a loader by source type, TextDocLoader is the default loader.
        loader_type = DocumentLoaderFactory.loader_mapping.get(source_type.lower(), TextDocLoader)

        return loader_type()
