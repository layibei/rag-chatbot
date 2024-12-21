import pytest
from datetime import datetime, UTC

from config.database.exceptions import EntityNotFoundError
from preprocess.index_log import IndexLog, Status, SourceType
from preprocess.index_log.repositories import IndexLogRepository

@pytest.fixture
def repository(db_manager):
    return IndexLogRepository(db_manager)

@pytest.fixture
def sample_log():
    return IndexLog(
        source="test.pdf",
        source_type=SourceType.PDF,
        checksum=f"abc123_{datetime.now(UTC).timestamp()}",
        status=Status.PENDING,
        created_at=datetime.now(UTC),
        created_by="test_user",
        modified_at=datetime.now(UTC),
        modified_by="test_user"
    )

class TestIndexLogRepository:
    def test_save_new_log(self, repository, sample_log):
        result = repository.save(sample_log)
        assert result.id is not None
        assert result.source == "test.pdf"
        assert result.status == Status.PENDING

    def test_save_existing_log(self, repository, sample_log):
        saved = repository.save(sample_log)
        saved.status = Status.COMPLETED
        updated = repository.save(saved)
        assert updated.status == Status.COMPLETED
        assert updated.id == saved.id

    def test_save_nonexistent_log(self, repository):
        non_existent = IndexLog(
            id=999,
            source="nonexistent.pdf",
            source_type=SourceType.PDF,
            checksum="xyz789",
            status=Status.PENDING,
            created_at=datetime.now(UTC),
            created_by="test_user",
            modified_at=datetime.now(UTC),
            modified_by="test_user"
        )
        with pytest.raises(EntityNotFoundError):
            repository.save(non_existent)

    def test_find_by_checksum(self, repository, sample_log):
        repository.save(sample_log)
        found = repository.find_by_checksum(sample_log.checksum)
        assert found is not None
        assert found.checksum == sample_log.checksum

    def test_find_by_source(self, repository, sample_log):
        repository.save(sample_log)
        found = repository.find_by_source(sample_log.source, sample_log.source_type)
        assert found is not None
        assert found.source == sample_log.source

    def test_delete_by_source(self, repository, sample_log):
        repository.save(sample_log)
        repository.delete_by_source(sample_log.source)
        found = repository.find_by_source(sample_log.source)
        assert found is None

    def test_get_next_pending_with_lock(self, repository, sample_log):
        repository.save(sample_log)
        next_pending = repository.get_next_pending_with_lock()
        assert next_pending is not None
        assert next_pending.status == Status.PENDING

    def test_find_by_id(self, repository, sample_log):
        saved = repository.save(sample_log)
        found = repository.find_by_id(saved.id)
        assert found is not None
        assert found.id == saved.id

    def test_create(self, repository):
        log = repository.create(
            source="new.pdf",
            source_type=SourceType.PDF,
            checksum="def456",
            status=Status.PENDING,
            user_id="test_user"
        )
        assert log.id is not None
        assert log.source == "new.pdf"
        assert log.created_by == "test_user"

    def test_list_logs(self, repository, sample_log):
        repository.save(sample_log)
        logs = repository.list_logs(page=1, page_size=10)
        assert len(logs) > 0
        assert isinstance(logs[0], IndexLog)

    def test_list_logs_with_search(self, repository, sample_log):
        repository.save(sample_log)
        logs = repository.list_logs(page=1, page_size=10, search="test")
        assert len(logs) > 0
        assert logs[0].source == "test.pdf" 