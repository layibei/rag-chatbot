import hashlib
import logging
import os.path
import traceback

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client.http import models

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
        if not documents or len(documents) < 1:
            return
        try:
            # For each document, check if it exists and what action should be taken
            ids = []
            docs = []
            for doc in documents:
                if not doc or not doc.page_content:
                    continue

                content = doc.page_content
                unique_key = self.generate_unique_key(content)
                metadatas = doc.metadata if doc.metadata is not None else {}
                metadatas["id"] = unique_key
                doc.set_metadata(metadatas)

                # Check if the document exists based on the unique key
                existing_doc = self.search_by_id(unique_key)

                if existing_doc:
                    # Compare the new document with the existing one
                    if existing_doc["page_content"] == content:
                        # No changes, so we ignore the update
                        self.logger.info(f"Document [{unique_key}] has no changes, skipping update.")
                        continue
                    else:
                        # Update the existing document
                        self.vectorstore.update(
                            id=unique_key,
                            values={"page_content": content, "metadata": metadatas}
                        )
                        self.logger.info(f"Updated document [{unique_key}].")
                else:
                    # The document does not exist, so we just add it
                    ids.append(unique_key)
                    docs.append(doc)
                    self.logger.info(f"Appending document [{unique_key}].")

                if len(ids) > 0:
                    self.vectorstore.add_documents(docs, ids=ids)
                    self.logger.info(f"Added {len(docs)} documents to vector store.")
        except AttributeError as e:
            self.logger.error(f"Error occured: {traceback.format_exc()}")

    @classmethod
    def generate_unique_key(self, text):
        # use doc content to generate SHA-256 hash
        hash_object = hashlib.sha256(text.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def search_by_metadata(self, metadata: dict) -> Document:
        self.logger.info(f"Searching for document with metadata: {dict}")

        if not metadata or not metadata.get("id"):
            return None

        # search by metadata id
        metadata_id = metadata["id"]
        metadata_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.id",
                    match=models.MatchValue(value=metadata_id)  # Replace with your document ID
                )
            ]
        )
        results = self.vectorstore.similarity_search_with_score(
            query="",
            k=1,
            filter=metadata_filter,
        )

        if len(results) > 0:
            self.logger.info(f"Found document with metadata id: {metadata_id}")
            return results[0]

    def search_by_id(self, id: str) -> Document:
        self.logger.info(f"Searching for document with id: {id}")
        results = self.vectorstore.search_by_ids([id])

        if len(results) > 0:
            self.logger.info(f"Found document with id: {id}")
            return results[0]