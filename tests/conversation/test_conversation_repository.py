import pytest
from datetime import datetime, UTC
from conversation.repositories import ConversationHistoryRepository
from conversation import ConversationHistory

@pytest.fixture
def repo(db_manager):
    return ConversationHistoryRepository(db_manager)

class TestConversationHistoryRepository:
    def test_save_conversation(self, repo):
        # Execute
        conversation = repo.create(ConversationHistory(
            user_id="test_user",
            session_id="test_session",
            request_id="test_request",
            user_input="test input",
            response="test response",
            created_at=datetime.now(UTC)
        ))

        # Assert
        assert conversation.id is not None

    def test_find_by_session(self, repo):
        # Setup
        for i in range(10):
            repo.create(ConversationHistory(
                user_id="test_user",
                session_id="test_session",
                request_id=f"request_{i}",
                user_input=f"input_{i}",
                response=f"response_{i}",
                created_at=datetime.now(UTC)
            ))

        # Execute
        history = repo.find_by_session("test_user", "test_session", limit=5)

        # Assert
        assert len(history) == 5
        assert all(h.user_id == "test_user" for h in history) 