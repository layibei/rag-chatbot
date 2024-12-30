import pytest
from unittest.mock import Mock
from handler.tools.response_formatter import ResponseFormatter

@pytest.fixture
def mock_llm():
    return Mock()

@pytest.fixture
def mock_config():
    return Mock()

@pytest.fixture
def formatter(mock_llm, mock_config):
    return ResponseFormatter(llm=mock_llm, config=mock_config)

def test_format_code_blocks(formatter):
    test_cases = [
        ("```public class Test {}```", "```public class Test {}```"),
        ("```def test():```", "```def test():```"),
        ("```<html>```", "```<html>```"),
        ("```random text```", "```random text```")
    ]

    for input_text, expected in test_cases:
        result = formatter._format_code_blocks(input_text)
        assert result.strip() == expected.strip()

def test_run_with_different_formats(formatter, mock_llm):
    test_cases = [
        ("json", "json"),
        ("markdown", "markdown"),
        ("code", "code"),
        ("table", "table")
    ]

    for format_type, expected_format in test_cases:
        mock_llm.invoke.return_value.content = format_type
        state = {
            "response": "test content",
            "documents": [],
            "user_input": "test query"
        }

        result = formatter.run(state)
        assert result["output_format"] == expected_format

def test_error_handling(formatter, mock_llm):
    mock_llm.invoke.side_effect = Exception("Test error")
    state = {
        "response": "test content",
        "documents": [],
        "user_input": "test query"
    }

    with pytest.raises(Exception, match="Test error"):
        formatter.run(state) 