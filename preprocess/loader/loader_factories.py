import logging
import os
from typing import Dict, Type

from preprocess.loader.base_loader import DocumentLoader
from preprocess.loader.csv_loader import CSVDocLoader
from preprocess.loader.json_loader import JsonDocLoader
from preprocess.loader.pdf_loader import PDFDocLoader
from preprocess.loader.text_loader import TextDocLoader

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    loader_mapping : Dict[str, Type[DocumentLoader]] = {
        '.pdf': PDFDocLoader,
        '.txt': TextDocLoader,
        '.csv': CSVDocLoader,
        '.json': JsonDocLoader,
    }

    # Factory method
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Invalid file path: {file_path}")
            raise ValueError(f"Invalid file path: {file_path}")

        file_suffix = os.path.splitext(file_path)[1]

        loader_type = DocumentLoaderFactory.loader_mapping.get(file_suffix, TextDocLoader)

        return loader_type()
