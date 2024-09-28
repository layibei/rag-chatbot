import asyncio
import hashlib
import os.path
import traceback
from datetime import datetime

import xxhash
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client.http import models

from loader.loader_factories import DocumentLoaderFactory
from preprocess import IndexLog
from preprocess.index_log_helper import IndexLogHelper
from utils.id_utils import get_id
from utils.logging_util import logger


class DocEmbeddingsProcessor:
    def __init__(self, embeddings: Embeddings, vectorstore: VectorStore, indexLogHelper: IndexLogHelper):
        self.logger = logger
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.indexLogHelper = indexLogHelper

    async def load_documents(self, dir_path: str):
        # Load documents from given dir_path
        if os.path.exists(dir_path):
            await self.__load(dir_path)
        else:
            self.logger.error(f"Directory {dir_path} does not exist.")

    async def __load(self, path: str):
        """
        load document from give path, the path might be a directory or a file, if it's a directory, then call itself to
        continue to iterate it
        """
        if os.path.isdir(path):
            files = os.listdir(path)
            await asyncio.gather(*[self.__load(os.path.join(path, file)) for file in files])
        else:
            # 0. check if filesize > 1Gb, if yes, print warning log and skip it
            if os.path.getsize(path) > 1024 * 1024 * 1024:
                self.logger.warning(f"File {path} is too large(>=1Gb), skip it.")
                return

            # 1.check if the file has been persisted or not
            log = self.indexLogHelper.find_by_source(path)
            # 2. if not, then load it and add it to vector store
            if not log:
                self.logger.info(f"Loading document: {path}")
                documents = await self.aload_document(path)
                self.logger.info(f"Loaded {len(documents)} documents from {path}")
                self.add_documents(documents)
                self.logger.info(f"Persisting index log for file: {path}")
                self.indexLogHelper.save(IndexLog(
                    id=get_id(),
                    source=path,
                    checksum=self.generate_xxhash64(path),
                    indexed_time=datetime.now().timestamp(),
                    indexed_by="system",
                    modified_time=datetime.now().timestamp(),
                    modified_by="system",
                ))
                self.logger.info(f"Persisted index log for file: {path}")
            # 3. if yes, then check if the content has changed or not, if not, by pass this file
            else:
                # hash on the file content
                self.logger.info(f"Document {path} has been indexed before, checking for changes.")
                # read content from file
                content = open(path, 'r').read()
                checksum = self.generate_xxhash64(content)
                # if not change on the content, then ignore
                if log.checksum == checksum:
                    self.logger.info(f"Document {path} has not been changed, ignore this file.")
                    return

                # if changes are detected, search for the document in vector store and delete those persisted docs to rebuild them
                self.logger.info(f"Document {path} has been indexed before.")
                searched_docs = self.search_by_metadata({"source": path, "checksum": checksum})
                if searched_docs is None or len(searched_docs) < 1:
                    self.logger.warning(f"Document {path} has been indexed before but not found in vector store.")
                # get ids from the searched documents
                ids = [doc.id for doc in searched_docs]
                self.vectorstore.delete(ids)

                # update the index log - checksum, last modified time
                log.set_checksum(checksum)
                log.set_modified_by("system")
                log.set_modified_time(datetime.now().timestamp())
                self.indexLogHelper.save(log)

                self.logger.info(f"Loading document: {path}")
                documents = await self.aload_document(path)
                self.logger.info(f"Persisting index log for file: {path}")
                self.add_documents(documents)
                self.logger.info(f"Persisted index log for file: {path}")

    async def aload_document(self, path: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, DocumentLoaderFactory.get_loader, path, path)

    def add_documents(self, documents, checksum: str):
        """
        Add documents to vector store
        :param documents: Documents loaded by file loader based on the file type.
        :param checksum: The original file checksum.
        """
        if not documents or len(documents) < 1:
            return
        try:
            ids = []
            docs = []
            # For each document, check if it exists and what action should be taken
            for doc in documents:
                if not doc or not isinstance(doc, Document):
                    self.logger.warning(
                        f"Skipping document: {doc} since it is empty or not a valid dictionary'.")
                    continue
                try:
                    content = doc.page_content
                    if not content:
                        self.logger.warning(f"Skipping document: {doc} since 'page_content' is empty.")
                        continue
                    # unique_key = self.generate_unique_key(content)
                    unique_key = self.generate_xxhash64(content)
                    metadata = doc.metadata if doc.metadata is not None else {}
                    metadata["key"] = unique_key
                    metadata["checksum"] = checksum

                    # The document does not exist, so we just add it
                    ids.append(unique_key)
                    docs.append(Document(page_content=content, metadata=metadata, id=unique_key))
                    self.logger.info(f"Appended document [{unique_key}].")
                except Exception as e:
                    self.logger.warning(f"Error occurred: {traceback.format_exc()}")

            if len(ids) > 0:
                self.vectorstore.add_documents(docs, ids=ids)
                self.logger.info(f"Added {len(docs)} documents to vector store.")
        except AttributeError as e:
            self.logger.error(f"Error occurred: {traceback.format_exc()}")

    @classmethod
    def generate_unique_key(self, text):
        # use doc content to generate SHA-256 hash
        hash_object = hashlib.sha256(text.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def generate_xxhash64(self, text) -> int:
        if not text:
            raise ValueError("Text cannot be empty")
        # Create an xxhash64 hasher
        hasher = xxhash.xxh64()
        # Update the hasher with the data
        hasher.update(text.encode('utf-8'))
        # Get the hash value
        hash_value = hasher.intdigest()
        self.logger.info(f"Generated xxhash64: {hash_value}")

        return hash_value

    def search_by_metadata(self, metadata: dict) -> Document:
        self.logger.info(f"Searching for document with metadata: {dict}")

        if not metadata:
            return None

        # search by metadata.checksum or metadata.source
        metadata_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=metadata.get("source", ""))  # Replace with your document ID
                ),
                models.FieldCondition(
                    key="metadata.checksum",
                    match=models.MatchValue(value=metadata.get("checksum", ""))  # Replace with your document ID
                )
            ]
        )
        results = self.vectorstore.similarity_search(
            query="",
            k=None,  # return all
            filter=metadata_filter,
        )
        self.logger.debug(f"Search results: {results}")

        return results

    def search_by_id(self, id: str) -> Document:
        self.logger.info(f"Searching for document with id: {id}")
        results = self.vectorstore.get_by_ids([id])

        if len(results) > 0:
            self.logger.info(f"Found document with id: {id}")
            return results[0]
