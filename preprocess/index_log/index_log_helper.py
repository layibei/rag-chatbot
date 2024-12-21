from typing import Optional, List

from sqlalchemy.exc import SQLAlchemyError

from config.database.exceptions import DatabaseError
from preprocess.index_log import IndexLog
from preprocess.index_log.repositories import IndexLogRepository
from utils.logging_util import logger


class IndexLogHelper:
    def __init__(self, repository: IndexLogRepository):
        self.repository = repository
        self.logger = logger

    def save(self, index_log: IndexLog) -> IndexLog:
        try:
            return self.repository.save(index_log)
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Unexpected error: {str(e)}")

    def find_by_checksum(self, checksum: str) -> Optional[IndexLog]:
        if not checksum:
            return None
        try:
            return self.repository.find_by_checksum(checksum)
        except SQLAlchemyError as e:
            self.logger.error(f'Error while finding index log for {checksum}')
            raise e

    def find_by_source(self, source: str, source_type: str = None) -> Optional[IndexLog]:
        if not source:
            return None
        try:
            return self.repository.find_by_source(source, source_type)
        except SQLAlchemyError as e:
            self.logger.error(f'Error while finding index log for {source}')
            raise e

    def delete_by_source(self, file_path: str):
        if not file_path:
            return None
        try:
            self.repository.delete_by_source(file_path)
            self.logger.info(f'Index log deleted for {file_path}')
        except SQLAlchemyError as e:
            self.logger.error(f'Error while deleting index log for {file_path}')
            raise e

    def get_pending_index_logs(self) -> List[IndexLog]:
        try:
            return self.repository.get_pending_index_logs()
        except Exception as e:
            self.logger.error(f"Error getting next pending document: {str(e)}")
            raise

    def find_by_id(self, log_id: int) -> Optional[IndexLog]:
        try:
            return self.repository.find_by_id(log_id)
        except Exception as e:
            self.logger.error(f"Error finding index log by id {log_id}: {str(e)}")
            raise

    def create(self, source: str, source_type: str, checksum: str, status: str, user_id: str) -> IndexLog:
        try:
            return self.repository.create(
                source=source,
                source_type=source_type,
                checksum=checksum,
                status=status,
                user_id=user_id
            )
        except SQLAlchemyError as e:
            self.logger.error('Error while creating index log')
            raise e

    def list_logs(self, page: int, page_size: int, search: Optional[str] = None) -> List[IndexLog]:
        try:
            return self.repository.list_logs(page, page_size, search)
        except SQLAlchemyError as e:
            self.logger.error('Error while listing logs')
            raise e
