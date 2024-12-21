import dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, UTC

from config.database.database_manager import DatabaseManager
from preprocess.index_log.index_log_helper import IndexLogHelper
from preprocess.index_log.repositories import IndexLogRepository
from utils.lock.distributed_lock_helper import DistributedLockHelper
from preprocess.index_log import Status
from preprocess.loader.loader_factories import DocumentLoaderFactory
from utils.lock.repositories import DistributedLockRepository
from utils.logging_util import logger
import hashlib
import os
import shutil
from pathlib import Path
from config.common_settings import CommonConfig
from typing import Optional

dotenv.load_dotenv(dotenv_path=os.getcwd() + "/.env")

db_manager = DatabaseManager(os.environ["POSTGRES_URI"])


class DocEmbeddingJob:
    def __init__(self):
        self.logger = logger
        self.config = CommonConfig()
        self.embeddings = self.config.get_model("embedding")
        self.vector_store = self.config.get_vector_store()

        index_log_repo = IndexLogRepository(db_manager)
        self.index_log_helper = IndexLogHelper(index_log_repo)
        self.distributed_lock_helper = DistributedLockHelper(DistributedLockRepository(db_manager))
        self.scheduler = BackgroundScheduler()
        self.setup_scheduler()

    def setup_scheduler(self):
        self.scheduler.add_job(
            self.process_pending_documents,
            'interval',
            minutes=3,
            id='process_pending_documents'
        )

        self.scheduler.add_job(
            self.scan_input_directory,
            'interval',
            minutes=3,
            id='scan_input_directory'
        )

        self.scheduler.start()

    def process_pending_documents(self):
        """Process one pending document at a time with distributed lock"""
        if not self.distributed_lock_helper.acquire_lock("process_pending_documents"):
            return

        try:
            logs = self.index_log_helper.get_pending_index_logs()
            if not logs:
                self.logger.info("No pending documents found")
                return

            for log in logs:
                try:
                    # Update status to IN_PROGRESS
                    log.status = Status.IN_PROGRESS
                    log.modified_at = datetime.now(UTC)
                    log.error_message = None
                    self.index_log_helper.save(log)

                    # Process document
                    self._process_document(log)

                    # Move file to archive folder
                    archive_path = self.config.get_embedding_config()["archive_path"]
                    os.makedirs(archive_path, exist_ok=True)

                    source_file = Path(log.source)
                    archive_file = Path(archive_path) / source_file.name

                    # Move the file
                    shutil.move(str(source_file), str(archive_file))

                    # Update source path in log
                    log.source = str(archive_file)

                    # Update status to COMPLETED
                    log.status = Status.COMPLETED
                    log.modified_at = datetime.now(UTC)
                    log.error_message = None
                    self.index_log_helper.save(log)
                except Exception as e:
                    log.status = Status.FAILED
                    log.error_message = str(e)
                    log.modified_at = datetime.now(UTC)
                    self.index_log_helper.save(log)
        finally:
            self.distributed_lock_helper.release_lock("process_pending_documents")

    def scan_input_directory(self):
        """Scan archive directory for new documents to process"""
        self.logger.info("Scanning input directory for new documents")
        if not self.distributed_lock_helper.acquire_lock("scan_input_directory"):
            self.logger.info("Failed to acquire lock, some other pod is processing the job.")
            return

        try:
            input_path = self.config.get_embedding_config()["input_path"]
            if not os.path.exists(input_path):
                self.logger.info(f"Archive directory does not exist: {input_path}")
                return

            for file_name in os.listdir(input_path):
                file_path = os.path.join(input_path, file_name)
                if not os.path.isfile(file_path):
                    continue

                self.logger.info(f"Processing file: {file_name}")

                try:
                    # Determine source type from file extension
                    file_extension = Path(file_name).suffix.lower()[1:]  # Remove the dot
                    source_type = self._get_source_type(file_extension)
                    if not source_type:
                        self.logger.warning(f"Unsupported file type: {file_name}")
                        continue

                    # Calculate checksum
                    checksum = self._calculate_checksum(file_path)

                    # Check if already processed
                    existing_log = self.index_log_helper.find_by_checksum(checksum)
                    if existing_log:
                        self.logger.info(f"Document already processed: {file_name}")
                        continue

                    # Create new index log
                    self.add_index_log(
                        source=file_path,
                        source_type=source_type,
                        user_id="system"  # System user for automated processing
                    )
                    self.logger.info(f"Added new document for processing: {file_name}")

                except Exception as e:
                    self.logger.error(f"Error processing input file {file_name}: {str(e)}")
                    continue

        finally:
            self.distributed_lock_helper.release_lock("scan_input_directory")

        self.logger.info("Finished scanning input directory for new documents")

    def _calculate_checksum(self, source: str) -> str:
        """Calculate checksum for a document"""
        try:
            with open(source, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {source}: {str(e)}")
            raise


    def add_index_log(self, source: str, source_type: str, user_id: str) -> dict:
        """Add a new document to the index log or update existing one"""
        # Calculate checksum from source file
        checksum = self._calculate_checksum(source)

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
        """Remove existing document embeddings from vector store"""
        docs = self.vector_store.search_by_metadata({
            "source": source,
            "checksum": checksum
        })
        if docs:
            self.vector_store.delete([doc.metadata["id"] for doc in docs])

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

    def _process_document(self, log):
        """Process a single document"""
        try:
            # Get appropriate loader
            loader = DocumentLoaderFactory.get_loader(log.source_type)

            # Load document
            documents = loader.load(log.source)

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": log.source,
                    "source_type": log.source_type,
                    "checksum": log.checksum
                })

            # Save to vector store
            self.vector_store.add_documents(documents)

            # Clear error message on success
            log.error_message = None

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise
