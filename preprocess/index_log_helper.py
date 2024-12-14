import traceback
from typing import Optional, List

from sqlalchemy import create_engine, select, and_, or_, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta, UTC

from preprocess import IndexLog
from utils.logging_util import logger
from . import Status

Base = declarative_base()


class IndexLogHelper():
    def __init__(self, uri: str):
        self.logger = logger
        self.engine = create_engine(uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_session(self):
        return self.Session()

    def save(self, index_log: IndexLog):
        with self.create_session() as session:
            try:
                # Create a new session-bound instance
                new_log = IndexLog(
                    id=index_log.id if hasattr(index_log, 'id') else None,
                    source=index_log.source,
                    source_type=index_log.source_type,
                    checksum=index_log.checksum,
                    status=index_log.status,
                    created_at=index_log.created_at,
                    created_by=index_log.created_by,
                    modified_at=index_log.modified_at,
                    modified_by=index_log.modified_by,
                    error_message=index_log.error_message if hasattr(index_log, 'error_message') else None
                )
                
                session.add(new_log)
                session.commit()
                
                # Update original instance with new values
                for attr, value in new_log.__dict__.items():
                    if not attr.startswith('_'):
                        setattr(index_log, attr, value)
                
                self.logger.info(f'Index log saved successfully, {index_log}')
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while saving index log,{traceback.format_exc()}')
                raise e

    def find_by_checksum(self, checksum: str) -> Optional[IndexLog]:
        if checksum is None:
            return None

        with self.create_session() as session:
            try:
                result = session.query(IndexLog).filter(IndexLog.checksum == checksum).first()
                if result:
                    # Create a detached copy of the result
                    log = IndexLog(
                        id=result.id,
                        source=result.source,
                        source_type=result.source_type,
                        checksum=result.checksum,
                        status=result.status,
                        created_at=result.created_at,
                        created_by=result.created_by,
                        modified_at=result.modified_at,
                        modified_by=result.modified_by,
                        error_message=result.error_message if hasattr(result, 'error_message') else None
                    )
                    return log
                return None
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {checksum}, {traceback.format_exc()}')
                raise e

    def find_by_source(self, source: str, source_type: str = None) -> Optional[IndexLog]:
        """Find index log by source and optionally source_type"""
        if source is None:
            return None

        with self.create_session() as session:
            try:
                query = session.query(IndexLog).filter(IndexLog.source == source)
                if source_type:
                    query = query.filter(IndexLog.source_type == source_type)
                
                result = query.first()
                if result:
                    # Create a detached copy of the result
                    log = IndexLog(
                        id=result.id,
                        source=result.source,
                        source_type=result.source_type,
                        checksum=result.checksum,
                        status=result.status,
                        created_at=result.created_at,
                        created_by=result.created_by,
                        modified_at=result.modified_at,
                        modified_by=result.modified_by,
                        error_message=result.error_message
                    )
                    return log
                return None
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {source}, {traceback.format_exc()}')
                raise e

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
                db_log = result.scalar_one_or_none()
                
                if db_log:
                    # Create a detached copy of the result
                    log = IndexLog(
                        id=db_log.id,
                        source=db_log.source,
                        source_type=db_log.source_type,
                        checksum=db_log.checksum,
                        status=db_log.status,
                        created_at=db_log.created_at,
                        created_by=db_log.created_by,
                        modified_at=db_log.modified_at,
                        modified_by=db_log.modified_by,
                        error_message=db_log.error_message if hasattr(db_log, 'error_message') else None
                    )
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
                result = session.query(IndexLog).get(log_id)
                if result:
                    # Create a detached copy of the result
                    log = IndexLog(
                        id=result.id,
                        source=result.source,
                        source_type=result.source_type,
                        checksum=result.checksum,
                        status=result.status,
                        created_at=result.created_at,
                        created_by=result.created_by,
                        modified_at=result.modified_at,
                        modified_by=result.modified_by,
                        error_message=result.error_message if hasattr(result, 'error_message') else None
                    )
                    return log
                return None
            except Exception as e:
                self.logger.error(f"Error finding index log by id {log_id}: {str(e)}")
                raise

    def create(self, source: str, source_type: str, checksum: str, status: str, user_id: str) -> IndexLog:
        with self.create_session() as session:
            try:
                # Create new log instance
                new_log = IndexLog(
                    source=source,
                    source_type=source_type,
                    checksum=checksum,
                    status=status,
                    created_at=datetime.now(UTC),
                    created_by=user_id,
                    modified_at=datetime.now(UTC),
                    modified_by=user_id
                )
                session.add(new_log)
                session.flush()  # Flush to get the ID and other DB-generated values
                
                # Create a detached copy before committing
                detached_log = IndexLog(
                    id=new_log.id,
                    source=new_log.source,
                    source_type=new_log.source_type,
                    checksum=new_log.checksum,
                    status=new_log.status,
                    created_at=new_log.created_at,
                    created_by=new_log.created_by,
                    modified_at=new_log.modified_at,
                    modified_by=new_log.modified_by
                )
                
                session.commit()
                return detached_log
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while creating index log: {traceback.format_exc()}')
                raise e

    def list_logs(self, page: int, page_size: int, search: Optional[str] = None) -> List[IndexLog]:
        with self.create_session() as session:
            query = session.query(IndexLog)
            
            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    or_(
                        IndexLog.source.ilike(search_pattern),
                        IndexLog.created_by.ilike(search_pattern),
                        IndexLog.modified_by.ilike(search_pattern)
                    )
                )
            
            return query.offset((page - 1) * page_size).limit(page_size).all()
