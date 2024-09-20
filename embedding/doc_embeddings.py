import logging
import os.path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from loader.loader_factories import DocumentLoaderFactory
from utils.logging_util import logger


class DocEmbeddings:
    def __init__(self, embeddings: Embeddings, vectorstore: VectorStore):
        self.logger = logger
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def load_documents(self, dir_path: str) -> VectorStore:
        # Load documents from given dir_path
        if os.path.exists(dir_path):
            self.__load(dir_path)
        else:
            self.logger.error(f"Directory {dir_path} does not exist.")

    def __load(self, path: str):
        """
        load document from give path, the path might be a directory or a file, if it's a directory, then call itself to
        continue to iterate it
        """
        if os.path.isdir(path):
            for file in os.listdir(path):
                self.__load(os.path.join(path, file))
        else:
            self.logger.info(f"Loading document: {path}")
            documents = DocumentLoaderFactory.get_loader(path).load(path)
            self.add_documents(documents)
            self.logger.info(f"Loaded {len(documents)} documents from {path}")

    def add_documents(self, documents):
        if not documents and len(documents) > 0:
            self.vectorstore.add_texts(
                texts=[doc.page_content for doc in documents],
                metadatas=[doc.metadata for doc in documents])
