from unittest.mock import Mock, patch

import pytest

from conversation import ConversationHistory


@pytest.fixture
def mock_dependencies():
    deps = {
        'config': Mock(),
        'llm': Mock(),
        'conversation_repo': Mock(),
        'llm_interaction_repo': Mock(),
        'vector_store': Mock()
    }
    # Setup conversation helper mock
    deps['conversation_repo'].create = Mock()
    # Setup LLM tracking mock
    deps['llm_interaction_repo'].create = Mock()
    return deps


@pytest.fixture
def query_handler(mock_dependencies):
    from handler.generic_query_handler import QueryHandler
    return QueryHandler(**mock_dependencies)


class TestQueryHandler:
    def test_handle_greeting(self, query_handler, mock_dependencies):
        # Setup
        mock_dependencies['llm'].predict.return_value = "true"
        test_input = "hello"
        test_request = "test_request"

        # Execute
        result = query_handler.handle(
            user_input=test_input,
            user_id="test_user",
            session_id="test_session",
            request_id=test_request
        )

        # Assert response
        assert result["data"] == "Hello! How can I help you today?"
        assert result["status"] == "success"
        assert result["user_input"] == test_input
        assert result["request_id"] == test_request
        
        # Verify LLM tracking
        assert mock_dependencies['llm_interaction_repo'].create.called
        
        # Verify conversation tracking
        assert mock_dependencies['conversation_repo'].create.call_count == 2

    def test_handle_normal_query(self, query_handler, mock_dependencies):
        # Setup
        mock_dependencies['llm'].predict.return_value = "false"
        mock_workflow_instance = Mock()
        mock_workflow_instance.invoke.return_value = "Test response"  # Ensure we return a string
        
        with patch('handler.workflow.query_process_workflow.QueryProcessWorkflow') as mock_workflow:
            mock_workflow.return_value = mock_workflow_instance  # Use the configured mock instance
            
            # Execute
            result = query_handler.handle(
                user_input="what is RAG?",
                user_id="test_user",
                session_id="test_session",
                request_id="test_request"
            )

            # Assert
            assert result["data"] == "Test response"
            assert result["status"] == "success"
            assert mock_dependencies['conversation_repo'].create.call_count == 2

    def test_error_handling(self, query_handler, mock_dependencies):
        # Setup
        test_error = Exception("Test error")
        mock_dependencies['llm'].predict.side_effect = test_error
        test_input = "test query"

        # Execute
        result = query_handler.handle(
            user_input=test_input,
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        # Assert
        assert result["status"] == "error"
        assert result["user_input"] == test_input
        assert "Test error" in result["error"]

    def test_greeting_check_error(self, query_handler, mock_dependencies):
        # Setup
        mock_dependencies['llm'].predict.side_effect = ValueError("Invalid input")
        
        # Execute
        result = query_handler.handle(
            user_input="hello",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        # Assert
        assert result["status"] == "error"
        assert "Invalid input" in result["error"]

    def test_conversation_tracking_error(self, query_handler, mock_dependencies):
        # Setup
        mock_dependencies['conversation_repo'].create.side_effect = Exception("DB Error")
        
        # Execute
        result = query_handler.handle(
            user_input="test",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request"
        )

        # Assert
        assert result["status"] == "error"
        assert "DB Error" in result["error"]

    def test_conversation_tracking(self, query_handler, mock_dependencies):
        # Setup
        mock_dependencies['llm'].predict.return_value = "false"
        with patch('handler.workflow.query_process_workflow.QueryProcessWorkflow') as mock_workflow:
            mock_workflow.return_value.invoke.return_value = "Test response"
            
            # Execute
            query_handler.handle(
                user_input="test query",
                user_id="test_user",
                session_id="test_session",
                request_id="test_request"
            )

            # Assert conversation tracking
            assert mock_dependencies['conversation_repo'].create.call_count == 2
            calls = mock_dependencies['conversation_repo'].create.call_args_list
            assert len(calls) == 2
            
            # Verify conversations
            for call in calls:
                conversation = call[0][0]
                assert isinstance(conversation, ConversationHistory)
                assert conversation.user_id == "test_user"
                assert conversation.session_id == "test_session"