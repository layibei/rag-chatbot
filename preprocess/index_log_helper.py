import traceback
from typing import Optional

from sqlalchemy import create_engine, select, and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta

from preprocess import IndexLog
from utils.logging_util import logger
from . import Status

Base = declarative_base()


class IndexLogHelper():
    def __init__(self, uri: str):
        self.logger = logger
        self.engine = create_engine(uri)
        Base.metadata.create_all(self.engine)

    def create_session(self):
        session = sessionmaker(bind=self.engine)
        return session()

    def save(self, index_log: IndexLog):

        with self.create_session() as session:
            try:
                session.add(index_log)
                session.commit()
                self.logger.info(f'Index log saved successfully, {index_log}')
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while saving index log,{traceback.format_exc()}')
                raise e

    def find_by_checksum(self, checksum: str) -> IndexLog:
        if checksum is None:
            return None

        with self.create_session() as session:
            try:
                log = session.query(IndexLog).filter(IndexLog.checksum == checksum).first()
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {checksum}, {traceback.format_exc()}')
                raise e

            return log

    def find_by_source(self, file_path: str) -> IndexLog:
        if file_path is None:
            return None

        with self.create_session() as session:
            try:
                log = session.query(IndexLog).filter(IndexLog.source == file_path).first()
                self.logger.info(f'Index log found for {file_path}, {log}')
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {file_path}, {traceback.format_exc()}')
                raise e

            return log

    def delete_by_source(self, file_path: str):
        if file_path is None:
            return None

        with self.create_session() as session:
            try:
                session.query(IndexLog).filter(IndexLog.source == file_path).delete()
                session.commit()
                self.logger.info(f'Index log deleted for {file_path}')
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while deleting index log for {file_path}, {traceback.format_exc()}')
                raise e

    def get_next_pending_with_lock(self) -> Optional[IndexLog]:
        """Get next pending document with distributed lock"""
        with self.create_session() as session:
            try:
                # Get one pending document with FOR UPDATE SKIP LOCKED
                stmt = select(IndexLog).where(
                    and_(
                        IndexLog.status == Status.PENDING,
                        # Avoid processing recently modified documents
                        IndexLog.modified_at < datetime.utcnow() - timedelta(minutes=5)
                    )
                ).with_for_update(skip_locked=True).limit(1)
                
                result = session.execute(stmt)
                log = result.scalar_one_or_none()
                
                if log:
                    session.commit()
                    return log
                return None
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error getting next pending document: {str(e)}")
                raise

    def find_by_id(self, log_id: int) -> Optional[IndexLog]:
        with self.create_session() as session:
            try:
                return session.query(IndexLog).get(log_id)
            except Exception as e:
                self.logger.error(f"Error finding index log by id {log_id}: {str(e)}")
                raise
