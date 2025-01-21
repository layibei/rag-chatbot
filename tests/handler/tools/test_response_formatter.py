import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage

from handler.tools.response_formatter import ResponseFormatter


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value.content = "markdown"
    return llm

@pytest.fixture
def mock_config():
    config = Mock()
    config.get_query_config.return_value = {
        "output": {
            "generate_suggested_documents": True
        }
    }
    return config

@pytest.fixture
def formatter(mock_llm, mock_config):
    return ResponseFormatter(llm=mock_llm, config=mock_config)

def test_format_code_blocks(formatter):
    test_cases = [
        ("```\ndef test():\n    pass\n```", "```python\ndef test():\n    pass\n```"),
        ("```\npublic class Test {}\n```", "```java\npublic class Test {}\n```"),
        ("```\n<html></html>\n```", "```html\n<html></html>\n```"),
        ("```\nrandom text\n```", "```plaintext\nrandom text\n```")
    ]

    for input_text, expected in test_cases:
        result = formatter._format_code_blocks(input_text)
        assert result.strip() == expected.strip()

# @pytest.mark.skip("Format detection needs to be fixed")
# def test_run_with_different_formats(formatter, mock_llm):
#     test_cases = [
#         (
#             "Here's a chart showing the data trends...\n```chart\nData visualization\n```",
#             "chart",
#             "formatted chart content",
#             "```chart\nformatted chart content\n```"
#         ),
#         (
#             "Here's a table of results...\n| Header1 | Header2 |\n|---------|----------|",
#             "table",
#             "formatted table content",
#             "```\nformatted table content\n```"
#         ),
#         (
#             "Here's some documentation...\n# Title\n## Subtitle",
#             "markdown",
#             "formatted markdown content",
#             "formatted markdown content"
#         ),
#         (
#             "Here's a code example:\n```python\ndef example():\n    pass\n```",
#             "code",
#             "formatted code content",
#             "```\nformatted code content\n```"
#         )
#     ]
#
#     for content, expected_format, llm_response, expected_output in test_cases:
#         mock_llm.invoke.reset_mock()
#         
#         def mock_invoke(messages, **kwargs):
#             prompt = messages[0].content if isinstance(messages[0], HumanMessage) else messages[0]
#             if "FORMAT CRITERIA" in prompt:
#                 return Mock(content=expected_format)
#             else:
#                 return Mock(content=llm_response)
#         
#         mock_llm.invoke.side_effect = mock_invoke
#         
#         state = {
#             "response": content,
#             "documents": [],
#             "user_input": "test query"
#         }
#
#         result = formatter.run(state)
#         assert result["output_format"] == expected_format, f"Failed for format: {expected_format}"
#         assert result["response"] == expected_output, f"Failed output for format: {expected_format}"
#         assert mock_llm.invoke.call_count == 2, f"Expected 2 LLM calls for {expected_format}"

def test_error_handling(formatter, mock_llm):
    mock_llm.invoke.side_effect = Exception("Test error")
    state = {
        "response": "test content",
        "documents": [],
        "user_input": "test query"
    }

    result = formatter.run(state)
    assert result["output_format"] == "markdown"  # Default fallback format
    assert result["response"] == "test content"  # Original response preserved

def test_detect_output_format(formatter):
    test_cases = [
        ("```json\n{\"key\": \"value\"}\n```", "markdown"),
        ("# Heading\n## Subheading", "markdown"),
        ("```python\ndef test():\n    pass\n```", "markdown"),
        ("| Header1 | Header2 |\n|----------|----------|\n| Cell1 | Cell2 |", "markdown")
    ]

    for content, expected_format in test_cases:
        format_type = formatter._detect_output_format(content, "test query")
        assert format_type == expected_format

def test_empty_input_handling(formatter):
    state = {
        "response": "",
        "documents": [],
        "user_input": ""
    }

    result = formatter.run(state)
    assert result == state  # Empty input returns original state 