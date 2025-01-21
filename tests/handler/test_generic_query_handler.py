import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, MagicMock

from handler.generic_query_handler import QueryHandler, QueryResponse
from conversation.repositories import ConversationHistoryRepository
from conversation.conversation_history_helper import ConversationHistoryHelper
from conversation import ConversationHistory


@pytest.fixture
def mock_dependencies():
    # Create mock objects
    db_manager = MagicMock()
    db_manager.session.return_value.__enter__.return_value = Mock()
    db_manager.session.return_value.__exit__.return_value = None

    # Other mock objects
    llm = Mock()
    vector_store = Mock()
    config = Mock()
    config.get_db_manager.return_value = db_manager

    # Create repository with mocked db_manager
    repository = ConversationHistoryRepository(db_manager)
    # Mock the save method
    repository.save = MagicMock()
    conversation_helper = ConversationHistoryHelper(repository)

    # Create a mock conversation history
    mock_conversation = ConversationHistory(
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

    # Configure the mock to return our conversation history
    db_manager.session.return_value.__enter__.return_value.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_conversation]

    return {
        'llm': llm,
        'vector_store': vector_store,
        'config': config,
        'db_manager': db_manager,
        'repository': repository,
        'conversation_helper': conversation_helper
    }


@pytest.fixture
def query_handler(mock_dependencies):
    handler = QueryHandler(
        llm=mock_dependencies['llm'],
        vector_store=mock_dependencies['vector_store'],
        config=mock_dependencies['config']
    )
    # Mock the logger
    handler.logger = Mock()
    handler.conversation_helper = mock_dependencies['conversation_helper']
    # Also mock the logger in conversation helper
    handler.conversation_helper.logger = Mock()
    return handler


class TestQueryHandler:
    def test_handle_greeting(self, query_handler, mock_dependencies):
        # Configure mock LLM responses
        mock_dependencies['llm'].invoke.side_effect = [
            Mock(content="GREETING"),
            Mock(content="Hello! How can I help you today?")
        ]

        result = query_handler.handle(
            user_input="hello",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        assert isinstance(result, QueryResponse)
        assert "Hello!" in result.answer
        assert isinstance(result.citations, list)
        assert isinstance(result.suggested_questions, list)
        assert isinstance(result.metadata, dict)

    def test_handle_domain_query(self, query_handler, mock_dependencies):
        # Configure mock responses
        mock_dependencies['llm'].invoke.side_effect = [
            Mock(content="DOMAIN_QUERY")
        ]

        # Mock workflow response
        query_handler._process_query = Mock(return_value={
            "answer": "Test answer",
            "citations": ["citation1"],
            "suggested_questions": ["question1"],
            "metadata": {}
        })

        result = query_handler.handle(
            user_input="what is AI?",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        assert result["answer"] == "Test answer"
        assert result["citations"] == ["citation1"]
        assert result["suggested_questions"] == ["question1"]

    def test_error_handling(self, query_handler, mock_dependencies):
        # Configure mock to raise an exception
        mock_dependencies['llm'].invoke.side_effect = Exception("Test error")

        with pytest.raises(Exception) as exc_info:
            query_handler.handle(
                user_input="test query",
                user_id="test_user",
                session_id="test_session",
                request_id="test_request"
            )

        assert str(exc_info.value) == "Test error"

    def test_conversation_tracking(self, query_handler, mock_dependencies):
        # Configure mock responses
        mock_dependencies['llm'].invoke.side_effect = [
            Mock(content="DOMAIN_QUERY")
        ]

        # Mock workflow response
        query_handler._process_query = Mock(return_value={
            "answer": "Test response",
            "citations": [],
            "suggested_questions": [],
            "metadata": {}
        })

        result = query_handler.handle(
            user_input="test query",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        # Verify conversation was saved
        mock_repository = mock_dependencies['conversation_helper'].repository
        assert mock_repository.save.called
        saved_conversation = mock_repository.save.call_args[0][0]
        assert saved_conversation.user_id == "test_user"
        assert saved_conversation.session_id == "test_session"
        assert "test query" in saved_conversation.user_input

    def test_unknown_query_type(self, query_handler, mock_dependencies):
        # Configure mock to return unknown query type
        mock_dependencies['llm'].invoke.side_effect = [
            Mock(content="UNKNOWN")
        ]

        # Mock workflow response
        query_handler._process_query = Mock(return_value={
            "answer": "Test response",
            "citations": [],
            "suggested_questions": [],
            "metadata": {}
        })

        result = query_handler.handle(
            user_input="unknown query",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        assert isinstance(result, dict)
        assert "answer" in result
        assert "citations" in result
        assert "suggested_questions" in result