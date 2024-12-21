import pytest
from unittest.mock import Mock
from datetime import datetime, UTC

from config.database.exceptions import DatabaseError
from preprocess.index_log import IndexLog, Status, SourceType
from preprocess.index_log.index_log_helper import IndexLogHelper

@pytest.fixture
def mock_repository():
    return Mock()

@pytest.fixture
def helper(mock_repository):
    return IndexLogHelper(mock_repository)

@pytest.fixture
def sample_log():
    return IndexLog(
        source="test.pdf",
        source_type=SourceType.PDF,
        checksum="abc123",
        status=Status.PENDING,
        created_at=datetime.now(UTC),
        created_by="test_user",
        modified_at=datetime.now(UTC),
        modified_by="test_user"
    )

class TestIndexLogHelper:
    def test_save(self, helper, mock_repository, sample_log):
        mock_repository.save.return_value = sample_log
        result = helper.save(sample_log)
        assert result == sample_log
        mock_repository.save.assert_called_once_with(sample_log)

    def test_save_error(self, helper, mock_repository, sample_log):
        mock_repository.save.side_effect = DatabaseError("Test error")
        with pytest.raises(DatabaseError):
            helper.save(sample_log)

    def test_find_by_checksum(self, helper, mock_repository, sample_log):
        mock_repository.find_by_checksum.return_value = sample_log
        result = helper.find_by_checksum("abc123")
        assert result == sample_log
        mock_repository.find_by_checksum.assert_called_once_with("abc123")

    def test_find_by_source(self, helper, mock_repository, sample_log):
        mock_repository.find_by_source.return_value = sample_log
        result = helper.find_by_source("test.pdf", SourceType.PDF)
        assert result == sample_log
        mock_repository.find_by_source.assert_called_once_with("test.pdf", SourceType.PDF)

    def test_delete_by_source(self, helper, mock_repository):
        helper.delete_by_source("test.pdf")
        mock_repository.delete_by_source.assert_called_once_with("test.pdf")

    def test_get_next_pending_with_lock(self, helper, mock_repository, sample_log):
        mock_repository.get_next_pending_with_lock.return_value = sample_log
        result = helper.get_next_pending_with_lock()
        assert result == sample_log
        mock_repository.get_next_pending_with_lock.assert_called_once()

    def test_find_by_id(self, helper, mock_repository, sample_log):
        mock_repository.find_by_id.return_value = sample_log
        result = helper.find_by_id(1)
        assert result == sample_log
        mock_repository.find_by_id.assert_called_once_with(1)

    def test_create(self, helper, mock_repository, sample_log):
        mock_repository.create.return_value = sample_log
        result = helper.create(
            source="test.pdf",
            source_type=SourceType.PDF,
            checksum="abc123",
            status=Status.PENDING,
            user_id="test_user"
        )
        assert result == sample_log
        mock_repository.create.assert_called_once()

    def test_list_logs(self, helper, mock_repository):
        mock_repository.list_logs.return_value = []
        result = helper.list_logs(page=1, page_size=10)
        assert result == []
        mock_repository.list_logs.assert_called_once_with(1, 10, None)

    def test_list_logs_with_search(self, helper, mock_repository):
        mock_repository.list_logs.return_value = []
        result = helper.list_logs(page=1, page_size=10, search="test")
        assert result == []
        mock_repository.list_logs.assert_called_once_with(1, 10, "test")