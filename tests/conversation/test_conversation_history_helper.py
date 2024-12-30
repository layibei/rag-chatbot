import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, patch

from conversation import ConversationHistory, ChatSession
from conversation.conversation_history_helper import ConversationHistoryHelper
from conversation.repositories import ConversationHistoryRepository


@pytest.fixture
def mock_repository():
    repository = Mock(spec=ConversationHistoryRepository)
    
    # Setup default return values
    repository.save.return_value = ConversationHistory(
        id="test_id",
        user_id="test_user",
        session_id="test_session",
        request_id="test_request",
        user_input="test input",
        response="test response",
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
        created_by="test_user",
        modified_by="test_user"
    )
    
    repository.find_by_session.return_value = [
        ConversationHistory(
            id=f"test_id_{i}",
            user_id="test_user",
            session_id="test_session",
            request_id=f"test_request_{i}",
            user_input=f"test input {i}",
            response=f"test response {i}",
            created_at=datetime.now(UTC),
            modified_at=datetime.now(UTC),
            created_by="test_user",
            modified_by="test_user"
        ) for i in range(3)
    ]
    
    repository.get_session_list.return_value = [
        ChatSession(
            session_id=f"session_{i}",
            title=f"Test Session {i}",
            last_message="test message",
            created_at=datetime.now(UTC)
        ) for i in range(2)
    ]
    
    return repository

@pytest.fixture
def helper(mock_repository):
    return ConversationHistoryHelper(mock_repository)

def test_save_conversation(helper, mock_repository):
    # Test saving a conversation
    result = helper.save_conversation(
        user_id="test_user",
        session_id="test_session",
        request_id="test_request",
        user_input="test input",
        response="test response"
    )
    
    # Verify the repository was called correctly
    mock_repository.save.assert_called_once()
    assert isinstance(result, ConversationHistory)
    assert result.user_id == "test_user"
    assert result.session_id == "test_session"
    assert result.request_id == "test_request"
    assert result.user_input == "test input"
    assert result.response == "test response"

def test_get_conversation_history(helper, mock_repository):
    # Test retrieving conversation history
    result = helper.get_conversation_history(
        user_id="test_user",
        session_id="test_session",
        limit=5
    )
    
    # Verify the repository was called correctly
    mock_repository.find_by_session.assert_called_once_with("test_user", "test_session", 5)
    assert len(result) == 3  # Based on mock data
    assert all(isinstance(r, ConversationHistory) for r in result)

def test_get_session_list(helper, mock_repository):
    # Test retrieving session list
    result = helper.get_session_list(user_id="test_user")
    
    # Verify the repository was called correctly
    mock_repository.get_session_list.assert_called_once_with("test_user")
    assert len(result) == 2  # Based on mock data
    assert all(isinstance(s, ChatSession) for s in result)
    assert all(s.title.startswith("Test Session") for s in result)

def test_update_message_like(helper, mock_repository):
    # Configure mock for update_message_like
    mock_repository.update_message_like.return_value = ConversationHistory(
        id="test_id",
        user_id="test_user",
        session_id="test_session",
        request_id="test_request",
        user_input="test input",
        response="test response",
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
        created_by="test_user",
        modified_by="test_user",
        liked=True
    )
    
    # Test updating message like status
    result = helper.update_message_like(
        user_id="test_user",
        session_id="test_session",
        request_id="test_request",
        liked=True
    )
    
    # Verify the repository was called correctly
    mock_repository.update_message_like.assert_called_once_with(
        user_id="test_user",
        session_id="test_session",
        request_id="test_request",
        liked=True
    )
    assert isinstance(result, ConversationHistory)
    assert result.liked is True

def test_delete_session(helper, mock_repository):
    # Configure mock for delete_session
    mock_repository.delete_session.return_value = True
    
    # Test deleting a session
    result = helper.delete_session(
        user_id="test_user",
        session_id="test_session"
    )
    
    # Verify the repository was called correctly
    mock_repository.delete_session.assert_called_once_with("test_user", "test_session")
    assert result is True

def test_error_handling(helper, mock_repository):
    # Configure mock to raise exception
    mock_repository.save.side_effect = Exception("Test error")
    
    # Test error handling in save_conversation
    with pytest.raises(Exception) as exc_info:
        helper.save_conversation(
            user_id="test_user",
            session_id="test_session",
            request_id="test_request",
            user_input="test input",
            response="test response"
        )
    
    assert str(exc_info.value) == "Test error" 