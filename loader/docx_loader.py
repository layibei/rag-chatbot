from langchain_community.document_loaders import CSVLoader, Docx2txtLoader
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from loader.base_loader import DocumentLoader


class DocxDocLoader(DocumentLoader):
    def get_loader(self, file_path: str) -> BaseLoader:
        if self.is_supported_file_extension(file_path):
            return Docx2txtLoader(file_path)

    def get_splitter(self, file_path: str) -> TextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)

    def is_supported_file_extension(self, file_path: str) -> bool:
        if None != file_path and file_path.endswith(".docx"):
            return True

        return False