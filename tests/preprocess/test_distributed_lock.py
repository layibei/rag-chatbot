import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from sqlalchemy.exc import IntegrityError
from utils.distributed_lock import DistributedLockHelper

@pytest.fixture
def mock_session():
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.query = Mock()
    return session

@pytest.fixture
def lock_helper(mock_session):
    session_factory = Mock(return_value=mock_session)
    return DistributedLockHelper(session_factory, "test_instance")

def test_acquire_lock_success(lock_helper, mock_session):
    result = lock_helper.acquire_lock("test_lock")
    assert result is True
    assert mock_session.add.called
    assert mock_session.commit.called

def test_acquire_lock_failure(lock_helper, mock_session):
    mock_session.commit.side_effect = IntegrityError(None, None, None)
    result = lock_helper.acquire_lock("test_lock")
    assert result is False
    assert mock_session.rollback.called

def test_release_lock(lock_helper, mock_session):
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    
    lock_helper.release_lock("test_lock")
    
    assert mock_query.delete.called
    assert mock_session.commit.called 