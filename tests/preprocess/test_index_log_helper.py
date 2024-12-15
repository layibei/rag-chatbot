import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, patch
from sqlalchemy.exc import SQLAlchemyError
from preprocess.index_log_helper import IndexLogHelper
from preprocess import IndexLog, Status

@pytest.fixture
def mock_session():
    session = Mock()
    session.query = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    return session

@pytest.fixture
def index_log_helper():
    with patch('sqlalchemy.create_engine'), \
         patch('sqlalchemy.orm.declarative_base'), \
         patch('sqlalchemy.orm.sessionmaker'):
        # Use SQLite for testing
        helper = IndexLogHelper('sqlite:///:memory:')
        return helper

def test_create_index_log(index_log_helper, mock_session):
    with patch.object(index_log_helper, 'create_session', return_value=mock_session):
        source = "test.pdf"
        source_type = "pdf"
        checksum = "abc123"
        status = Status.PENDING
        user_id = "test_user"
        
        result = index_log_helper.create(source, source_type, checksum, status, user_id)
        
        assert result.source == source
        assert result.source_type == source_type
        assert result.checksum == checksum
        assert result.status == status
        assert result.created_by == user_id
        assert mock_session.add.called
        assert mock_session.commit.called

def test_find_by_checksum(index_log_helper, mock_session):
    mock_log = Mock(spec=IndexLog)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_log
    
    with patch.object(index_log_helper, 'create_session', return_value=mock_session):
        result = index_log_helper.find_by_checksum("test_checksum")
        assert result is not None
        mock_session.query.assert_called_once()

def test_save_index_log(index_log_helper, mock_session):
    test_log = IndexLog(
        source="test.pdf",
        source_type="pdf",
        checksum="abc123",
        status=Status.PENDING,
        created_at=datetime.now(UTC),
        created_by="test_user",
        modified_at=datetime.now(UTC),
        modified_by="test_user"
    )
    
    with patch.object(index_log_helper, 'create_session', return_value=mock_session):
        index_log_helper.save(test_log)
        assert mock_session.add.called
        assert mock_session.commit.called 