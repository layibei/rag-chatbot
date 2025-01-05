import pytest
from unittest.mock import Mock, patch
from handler.tools.response_grader import ResponseGrader

@pytest.fixture
def mock_llm():
    return Mock()

@pytest.fixture
def mock_config():
    config = Mock()
    config.get_query_config.return_value = 0.7
    return config

@pytest.fixture
def grader(mock_llm, mock_config):
    return ResponseGrader(mock_llm, mock_config)

def test_response_grading_good_answer(grader, mock_llm):
    mock_llm.invoke.return_value = Mock(content="0.85")
    
    result = grader.run({
        "response": "Python 3.9 was released in October 2020.",
        "user_input": "When was Python 3.9 released?"
    })
    
    assert result["needs_rewrite"] is False
    assert result["response_grade"] == 0.85

def test_response_grading_poor_answer(grader, mock_llm):
    mock_llm.invoke.return_value = Mock(content="0.45")
    
    result = grader.run({
        "response": "Python is a programming language.",
        "user_input": "When was Python 3.9 released?"
    })
    
    assert result["needs_rewrite"] is True
    assert result["response_grade"] == 0.45 