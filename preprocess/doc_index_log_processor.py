import hashlib
import os
from datetime import datetime, UTC
from typing import Optional

import dotenv
from langchain_postgres import PGVector
from langchain_redis import RedisVectorStore

from config.common_settings import CommonConfig
from config.database.database_manager import DatabaseManager
from preprocess.index_log import Status
from utils.logging_util import logger

dotenv.load_dotenv(dotenv_path=os.getcwd()+"/.env")


class DocEmbeddingsProcessor:
    def __init__(self, embeddings, vector_store, index_log_helper):
        self.logger = logger
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.index_log_helper = index_log_helper
        self.config = CommonConfig()

    def _get_source_type(self, extension: str) -> Optional[str]:
        """Map file extension to source type"""
        extension_mapping = {
            'pdf': 'pdf',
            'txt': 'text',
            'csv': 'csv',
            'json': 'json',
            'docx': 'docx'
        }
        return extension_mapping.get(extension.lower())

    def add_index_log(self, source: str, source_type: str, user_id: str) -> dict:
        """Add a new document to the index log or update existing one"""
        # Calculate checksum from source file
        checksum = self._calculate_checksum(source)

        # First check by checksum
        existing_log = self.index_log_helper.find_by_checksum(checksum)
        if existing_log:
            return {
                "message": "Document with same content already exists",
                "source": existing_log.source,
                "source_type": existing_log.source_type,
                "id": existing_log.id
            }

        # Then check by source path
        existing_log = self.index_log_helper.find_by_source(source, source_type)
        if existing_log:
            # Content changed, update existing log
            self._remove_existing_embeddings(source, existing_log.checksum)
            existing_log.checksum = checksum
            existing_log.status = Status.PENDING
            existing_log.modified_at = datetime.now(UTC)
            existing_log.modified_by = user_id
            self.index_log_helper.save(existing_log)
            return {
                "message": "Document updated and queued for processing",
                "id": existing_log.id,
                "source": source,
                "source_type": source_type
            }

        # Create new log
        new_log = self.index_log_helper.create(
            source=source,
            source_type=source_type,
            checksum=checksum,
            status=Status.PENDING,
            user_id=user_id
        )
        return {
            "message": "Document queued for processing",
            "id": new_log.id,
            "source": source,
            "source_type": source_type
        }

    def _calculate_checksum(self, source: str) -> str:
        """Calculate checksum for a document"""
        try:
            with open(source, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {source}: {str(e)}")
            raise

    def _remove_existing_embeddings(self, source: str, checksum: str):
        """Remove existing document embeddings from vector store

        Args:
            source (str): The identifier of the document source
            checksum (str): The checksum of the document, used to uniquely identify the document
        """
        # Search for document embeddings that match the source and checksum in the vector store
        if isinstance(self.vector_store, RedisVectorStore):
            docs = self.vector_store.search_by_metadata({
                "source": source,
                "checksum": checksum
            })
        elif isinstance(self.vector_store, PGVector):
            # Create a filter condition for metadata
            filter = {
                "metadata": {
                    "source": source,
                    "checksum": checksum
                }
            }
            # Use a dummy vector since we only care about metadata
            dummy_vector = [0.0] * self.embeddings.dimension
            docs = self.vector_store.similarity_search_by_vector(
                embedding=dummy_vector,
                k=100,  # Adjust based on expected max documents
                filter=filter
            )

        # If matching document embeddings are found, delete them from the vector store
        if docs:
            self.vector_store.delete([doc.metadata["id"] for doc in docs])


    def get_document_by_id(self, log_id):
        """Get document by ID"""
        log = self.index_log_helper.find_by_id(log_id)
        return log
