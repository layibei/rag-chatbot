import asyncio
import hashlib
import os.path
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict
from sqlalchemy.exc import IntegrityError

import xxhash
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisVectorStore
from qdrant_client.http import models

from config.common_settings import CommonConfig
from preprocess.loader import DocumentLoaderFactory
from preprocess import IndexLog, Status, SourceType
from preprocess.index_log_helper import IndexLogHelper
from utils.date_util import get_timestamp_in_utc
from utils.id_util import get_id
from utils.logging_util import logger


class DocEmbeddingsProcessor:
    def __init__(self, embeddings: Embeddings, vectorstore: VectorStore, index_log_helper: IndexLogHelper):
        self.logger = logger
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.index_log_helper = index_log_helper
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.base_config = CommonConfig()

    async def load_documents(self, dir_path: str):
        """
        Load documents from given path.
        :param dir_path:
        :return:
        """
        # Load documents from given dir_path
        if os.path.exists(dir_path):
            self.logger.info(f"Loading documents from directory: {dir_path}")
            await self.__load(dir_path)
        else:
            self.logger.error(f"Directory {dir_path} does not exist.")
            return;

        self.logger.info("document pre-process are done.")

    async def __load(self, path: str):
        """
        load document from give path, the path might be a directory or a file, if it's a directory, then call itself to
        continue to iterate it
        """
        # check if the file directory is empty
        if not os.listdir(path):
            self.logger.info(f"Directory {path} is empty, will by pass the document preprocessing.")
            return

        if os.path.isdir(path):
            files = os.listdir(path)
            await asyncio.gather(*[self.__load(os.path.join(path, file)) for file in files])
        else:
            # 0. check if filesize > 1Gb, if yes, print warning log and skip it
            if os.path.getsize(path) > 1024 * 1024 * 1024:
                self.logger.warning(f"File {path} is too large(>=1Gb), skip it.")
                return

            # read content from file
            content = open(path, mode='r', encoding='utf-8', errors='ignore').read()
            checksum = self.generate_xxhash64(content)
            # 1.check if the file has been persisted or not
            log = self.index_log_helper.find_by_source(path)
            # 2. if not, then load it and add it to vector store
            if not log:
                self.logger.info(f"Loading document: {path}")
                documents = await self.aload_document(path)
                self.logger.info(f"Loaded {len(documents)} documents from {path}")
                self.add_documents(documents, checksum)
                self.logger.info(f"Persisting index log for file: {path}")
                self.index_log_helper.save(IndexLog(
                    # id=get_id(),
                    source=path,
                    checksum=self.generate_xxhash64(path),
                    indexed_time=get_timestamp_in_utc(),
                    indexed_by="system",
                    modified_time=get_timestamp_in_utc(),
                    modified_by="system",
                    status="COMPLETED"
                ))
                self.logger.info(f"Persisted index log for file: {path}")
            # 3. if yes, then check if the content has changed or not, if not, by pass this file
            else:
                # hash on the file content
                self.logger.info(f"Document {path} has been indexed before, checking for changes.")

                # if not change on the content, then ignore
                if log.checksum == str(checksum):
                    self.logger.info(f"Document {path} has not been changed, ignore this file.")
                    return

                # if changes are detected, search for the document in vector store and delete those persisted docs to rebuild them
                self.logger.info(f"Document {path} has been indexed before, checksum:{checksum}.")
                searched_docs = self.search_by_metadata({"source": path, "checksum": checksum})
                if searched_docs is None or len(searched_docs) < 1:
                    self.logger.warning(f"Document {path} has been indexed before but not found in vector store.")
                # get ids from the searched documents
                ids = [doc.id for doc in searched_docs]
                self.vectorstore.delete(ids)

                # update the index log - checksum, last modified time
                log.checksum = checksum
                log.modified_by = "system"
                log.modified_time = get_timestamp_in_utc()
                log.status = "COMPLETED"
                self.index_log_helper.save(log)

                self.logger.info(f"Loading document: {path}")
                documents = await self.aload_document(path)
                self.logger.info(f"Persisting index log for file: {path}")

                # set id for documents if it does not exist, document should has id attribute:str
                for doc in documents:
                    if not doc.id:
                        doc.id = get_id()

                self.add_documents(documents, checksum)
                self.logger.info(f"Persisted index log for file: {path}")
            # move file to archive folder
            self.logger.info(f"Moving file: {path} to archive folder.")
            directory, filename = os.path.split(path)
            os.rename(path, os.path.join(self.base_config.get_embedding_config().get("archive_path"), filename))

    async def aload_document(self, path: str):
        loop = asyncio.get_event_loop()
        # get loader from DocumentLoaderFactory by file
        loader = DocumentLoaderFactory.get_loader(path)
        # then load the document
        return await loop.run_in_executor(self.executor, loader.load, path)

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
                    metadata["checksum"] = unique_key
                    metadata["parent_checksum"] = checksum

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

        results = None
        # search by metadata.parent_checksum or metadata.source
        # Construct the query safely
        source_value = metadata.get('source', '')
        parent_checksum_value = metadata.get('parent_checksum', '')

        # Sanitize inputs to prevent injection attacks
        source_value = source_value.replace("'", "\\'")
        parent_checksum_value = parent_checksum_value.replace("'", "\\'")
        if isinstance(self.vectorstore, QdrantVectorStore):
            metadata_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_value)  # Replace with your document ID
                    ),
                    models.FieldCondition(
                        key="metadata.parent_checksum",
                        match=models.MatchValue(value=parent_checksum_value)
                        # Replace with your document ID
                    )
                ]
            )
            results = self.vectorstore.similarity_search(
                query="",
                k=None,  # return all
                filter=metadata_filter,
            )

        elif isinstance(self.vectorstore, RedisVectorStore):
            # Apply filter using Redis Search Query
            query = f"@source:{source_value} @parent_checksum:{parent_checksum_value}"
            results = self.vectorstore.similarity_search(query)

        self.logger.debug(f"Search results: {results}")

        return results

    def search_by_id(self, id: str) -> Document:
        self.logger.info(f"Searching for document with id: {id}")
        results = self.vectorstore.get_by_ids([id])

        if len(results) > 0:
            self.logger.info(f"Found document with id: {id}")
            return results[0]

    def add_index_log(self, source: str, source_type: SourceType) -> Dict:
        try:
            # Calculate checksum of content
            content = self._load_content(source)
            checksum = str(self.generate_xxhash64(content))

            # Check if document already exists
            existing_log = self.index_log_helper.find_by_source(source)
            
            if existing_log:
                if existing_log.checksum == checksum:
                    return {
                        "message": "the source has already been processed",
                        "source": source,
                        "source_type": source_type
                    }

            # Create new index log
            index_log = IndexLog(
                source=source,
                source_type=source_type,
                checksum=checksum,
                created_at=datetime.utcnow(),
                indexed_by="system",
                modified_at=datetime.utcnow(),
                modified_by="system",
                status=Status.PENDING
            )
            
            self.index_log_helper.save(index_log)
            return {
                "id": index_log.id,
                "source": source,
                "source_type": source_type
            }

        except Exception as e:
            self.logger.error(f"Error adding index log: {str(e)}")
            raise

    def get_status(self, log_id: int) -> Dict:
        log = self.index_log_helper.find_by_id(log_id)
        if not log:
            raise ValueError(f"Index log {log_id} not found")
        
        return {
            "id": log.id,
            "status": log.status,
            "error_message": log.error_message
        }

    def process_pending_documents(self):
        """Process one pending document at a time"""
        try:
            # Get one pending document with lock
            log = self.index_log_helper.get_next_pending_with_lock()
            if not log:
                return

            try:
                # Update status to IN_PROGRESS
                log.status = Status.IN_PROGRESS
                log.modified_at = datetime.utcnow()
                self.index_log_helper.save(log)

                # Process document
                self._process_document(log)
                
                # Update status to COMPLETED
                log.status = Status.COMPLETED
                log.modified_at = datetime.utcnow()
                self.index_log_helper.save(log)

            except Exception as e:
                log.status = Status.FAILED
                log.error_message = str(e)
                log.modified_at = datetime.utcnow()
                self.index_log_helper.save(log)
                raise

        except Exception as e:
            self.logger.error(f"Error processing pending documents: {str(e)}")
            raise

    def _process_document(self, index_log: IndexLog):
        """Process a single document and store in vector database"""
        try:
            # Load and parse document
            loader = DocumentLoaderFactory.get_loader(index_log.source)
            documents = loader.load(index_log.source)

            # Add documents to vector store
            self.add_documents(documents, index_log.checksum)

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise
