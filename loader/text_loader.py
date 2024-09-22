from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.document_loaders import BaseLoader

from loader.base_loader import DocumentLoader


class TextDocLoader(DocumentLoader):
    def get_splitter(self, file_path):
        return RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)

    def get_loader(self, file_path: str) -> BaseLoader:
        if self.is_supported_file_extension(file_path):
            return TextLoader(file_path)

        return None

    def is_supported_file_extension(self, file_path: str) -> bool:
        # This is default loader, does not check the file extension here.
        return True
