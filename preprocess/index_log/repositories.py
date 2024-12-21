from datetime import datetime, UTC, timedelta
from typing import Optional, List

from sqlalchemy import and_, or_

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
                    existing = session.get(self.model_class, index_log.id)
                    if existing:
                        # Update existing record
                        for attr, value in index_log.__dict__.items():
                            if not attr.startswith('_') and attr != 'id':
                                setattr(existing, attr, value)
                        session.merge(existing)
                    else:
                        # Add as new if not found
                        session.add(index_log)

                session.flush()
                session.refresh(index_log)
                return self._create_detached_copy(index_log)
            except Exception as e:
                session.rollback()
                if "UNIQUE constraint failed" in str(e):
                    # Handle unique constraint violation
                    existing = session.query(self.model_class) \
                        .filter_by(checksum=index_log.checksum) \
                        .first()
                    if existing:
                        # Update existing record
                        for attr, value in index_log.__dict__.items():
                            if not attr.startswith('_') and attr != 'id':
                                setattr(existing, attr, value)
                        session.merge(existing)
                        session.flush()
                        session.refresh(existing)
                        return self._create_detached_copy(existing)
                raise

    def find_by_checksum(self, checksum: str) -> Optional[IndexLog]:
        result = self.find_by_filter(checksum=checksum)
        if not result or len(result) < 1:
            return None
        return self._create_detached_copy(result.first) if result else None

    def find_by_source(self, source: str, source_type: str = None) -> Optional[IndexLog]:
        filters = {"source": source}
        if source_type:
            filters["source_type"] = source_type
        result = self.find_by_filter(**filters)
        if not result or len(result) < 1:
            return None
        return self._create_detached_copy(result.first()) if result else None

    def delete_by_source(self, file_path: str) -> None:
        with self.db_manager.session() as session:
            session.query(self.model_class).filter(
                self.model_class.source == file_path
            ).delete()

    def get_pending_index_logs(self) -> List[IndexLog]:
        with self.db_manager.session() as session:
            stmt = session.query(self.model_class).filter(
                and_(
                    self.model_class.status == Status.PENDING
                )
            ).with_for_update(skip_locked=True)

            results = stmt.all()
            return [self._create_detached_copy(result) for result in results]

    def find_by_id(self, log_id: int) -> Optional[IndexLog]:
        result = self.find_by_filter(first=True, id=log_id)
        return self._create_detached_copy(result) if result else None

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
        query = self.query()

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
