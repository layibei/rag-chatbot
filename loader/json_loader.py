import json
import os
from typing import List

from langchain_community.document_loaders import JSONLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters.json import RecursiveJsonSplitter

from loader.base_loader import DocumentLoader


class JsonDocLoader(DocumentLoader):

    def load(self, file_path: str) -> List[Document]:
        # Input validation
        if not file_path or not os.path.exists(file_path):
            self.logger.error(f"Invalid file path: {file_path}")
            raise ValueError(f"Invalid file path: {file_path}")

        try:
            with open(file_path, 'r') as file:
                document = json.load(file)

            if not document:
                self.logger.error(f"Loaded document content is empty: {file_path}")
                raise ValueError(f"Loaded document content is empty: {file_path}")

            splitter = self.get_splitter(document)
            if not splitter:
                self.logger.error(f"Failed to create splitter for file: {file_path}")
                raise ValueError(f"Failed to create splitter for file: {file_path}")

            return splitter.split_json(document)
        except Exception as e:
            self.logger.error(f"Failed to load document: {file_path}, Error: {str(e)}")
            raise

    def get_loader(self, file_path: str) -> BaseLoader:
        pass

    def get_splitter(self, file_path: str):
        return RecursiveJsonSplitter()

    def is_supported_file_extension(self, file_path: str) -> bool:
        if None != file_path and file_path.endswith(".json"):
            return True

        return False
