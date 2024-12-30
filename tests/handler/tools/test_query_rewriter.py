import pytest
from handler.tools.query_rewriter import QueryRewriter

@pytest.fixture
def rewriter(mock_llm):
    return QueryRewriter(mock_llm)

class TestQueryRewriter:
    def test_successful_rewrite(self, rewriter, mock_llm):
        mock_llm.invoke.return_value.content = "What is the current stable version of Python as of 2024?"
        state = {"user_input": "What's the latest Python version?"}
        
        result = rewriter.run(state)
        assert isinstance(result, str)
        assert "Python" in result
        assert "2024" in result

    def test_validation_failure(self, rewriter, mock_llm):
        mock_llm.invoke.return_value.content = "What is the latest Java version?"  # Different topic
        state = {"user_input": "What is Python?"}
        
        result = rewriter.run(state)
        assert result == "What is Python?"  # Should return original

    def test_empty_input_handling(self, rewriter):
        state = {"user_input": ""}
        result = rewriter.run(state)
        assert result == ""

    def test_error_handling(self, rewriter, mock_llm):
        mock_llm.invoke.side_effect = Exception("Test error")
        state = {"user_input": "test query"}
        result = rewriter.run(state)
        assert result == "test query"  # Should return original

    def test_validate_rewrite(self, rewriter):
        test_cases = [
            ("Python version", "Latest Python version 3.12", True),
            ("Java features", "Common Java programming features", True),
            ("Docker commands", "Basic Docker CLI commands", True),
            ("weather in paris", "temperature in london", False)
        ]
        
        for original, rewritten, expected in test_cases:
            assert rewriter._validate_rewrite(original, rewritten) == expected 