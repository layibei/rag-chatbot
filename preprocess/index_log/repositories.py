from datetime import datetime, UTC, timedelta
from typing import Optional, List

from sqlalchemy import and_, or_
from sqlalchemy.sql import select

from config.database.repository import BaseRepository
from preprocess.index_log import IndexLog, Status
from utils.id_util import get_id


class IndexLogRepository(BaseRepository[IndexLog]):
    def __init__(self, db_manager):
        super().__init__(db_manager, IndexLog)

    def _get_model_class(self) -> type:
        return IndexLog

    def save(self, index_log: IndexLog) -> IndexLog:
        with self.db_manager.session() as session:
            try:
                if not index_log.id:
                    index_log.id = get_id()
                    session.add(index_log)
                else:
                    index_log = session.merge(index_log)

                session.flush()
                session.refresh(index_log)
                return self._create_detached_copy(index_log)

            except Exception as e:
                session.rollback()
                raise

    def find_by_checksum(self, checksum: str) -> Optional[IndexLog]:
        results = self.find_by_filter(checksum=checksum)
        return results[0] if results else None

    def find_by_source(self, source: str, source_type: str = None) -> Optional[IndexLog]:
        filters = {"source": source}
        if source_type:
            filters["source_type"] = source_type
        result = self.find_by_filter(**filters)
        if not result or len(result) < 1:
            return None
        return self._create_detached_copy(result[0]) if result else None

    def delete_by_source(self, file_path: str) -> None:
        with self.db_manager.session() as session:
            session.query(self.model_class).filter(
                self.model_class.source == file_path
            ).delete()

    def get_pending_index_logs(self) -> List[IndexLog]:
        with self.db_manager.session() as session:
            stmt = session.query(self.model_class).filter(
                or_(
                    self.model_class.status == Status.PENDING,
                    and_(
                        self.model_class.status == Status.FAILED,
                        self.model_class.retry_count <= 3
                    )
                )
            ).with_for_update(skip_locked=True)

            results = stmt.all()
            return [self._create_detached_copy(result) for result in results]

    def find_by_id(self, log_id: str) -> Optional[IndexLog]:
        result = self.find_by_filter(id=log_id)
        return result

    def create(self, source: str, source_type: str, checksum: str, status: Status, user_id: str) -> IndexLog:
        with self.db_manager.session() as session:
            try:
                now = datetime.now(UTC)
                index_log = IndexLog(
                    id=get_id(),
                    source=source,
                    source_type=source_type,
                    checksum=checksum,
                    status=status,
                    created_at=now,
                    created_by=user_id,
                    modified_at=now,
                    modified_by=user_id
                )
                session.add(index_log)
                session.flush()
                session.refresh(index_log)
                return self._create_detached_copy(index_log)
            except Exception as e:
                session.rollback()
                if "UNIQUE constraint failed" in str(e):
                    existing = session.query(self.model_class) \
                        .filter_by(checksum=checksum) \
                        .first()
                    if existing:
                        return self._create_detached_copy(existing)
                raise

    def list_logs(self, page: int, page_size: int, search: Optional[str] = None) -> List[IndexLog]:
        with self.db_manager.session() as session:
            query = session.query(self.model_class)

            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    or_(
                        self.model_class.source.ilike(search_pattern),
                        self.model_class.created_by.ilike(search_pattern),
                        self.model_class.modified_by.ilike(search_pattern)
                    )
                )

            results = query.offset((page - 1) * page_size).limit(page_size).all()
            return [self._create_detached_copy(result) for result in results]

    def _create_detached_copy(self, db_obj: Optional[IndexLog]) -> Optional[IndexLog]:
        if not db_obj:
            return None

        return IndexLog(
            id=db_obj.id,
            source=db_obj.source,
            source_type=db_obj.source_type,
            checksum=db_obj.checksum,
            status=db_obj.status,
            created_at=db_obj.created_at,
            created_by=db_obj.created_by,
            modified_at=db_obj.modified_at,
            modified_by=db_obj.modified_by,
            error_message=db_obj.error_message
        )

    def find_all(self, filters: dict):
        query = select(IndexLog)
        
        if 'status' in filters:
            query = query.where(IndexLog.status == filters['status'])
        
        if 'modified_at_lt' in filters:
            query = query.where(IndexLog.modified_at < filters['modified_at_lt'])
            
        with self.db_manager.session() as session:
            result = session.execute(query)
            results = result.scalars().all()
            return [self._create_detached_copy(result) for result in results]

    def query(self):
        """Create a new query object for the model class."""
        with self.db_manager.session() as session:
            return session.query(self.model_class)
