import pytest
from unittest.mock import Mock, patch
from handler.workflow.query_process_workflow import QueryProcessWorkflow
from langchain_core.documents import Document
from langchain_postgres import PGVector
import logging
from langchain.schema import HumanMessage

@pytest.fixture
def mock_logger():
    with patch('handler.workflow.query_process_workflow.logger') as mock_log, \
         patch('handler.tools.document_retriever.logger') as mock_doc_log, \
         patch('handler.tools.web_search_tool.logger') as mock_web_log, \
         patch('handler.tools.response_formatter.logger') as mock_fmt_log, \
         patch('handler.tools.query_rewriter.logger') as mock_rewrite_log, \
         patch('handler.tools.hallucination_detector.logger') as mock_hall_log:
        
        # Configure each mock logger to handle the metadata field
        loggers = [mock_log, mock_doc_log, mock_web_log, mock_fmt_log, 
                  mock_rewrite_log, mock_hall_log]
        
        for logger in loggers:
            logger.info = Mock()
            logger.debug = Mock()
            logger.warning = Mock()
            logger.error = Mock()
            
            # Add metadata filter to match logging_util.py format
            logger.filters = []
            
            class MetadataFilter(logging.Filter):
                def filter(self, record):
                    if not hasattr(record, 'metadata'):
                        record.metadata = ''
                    return True
            
            logger.addFilter(MetadataFilter())
        
        yield mock_log

@pytest.fixture
def mock_vector_store():
    vector_store = Mock(spec=PGVector)
    vector_store.similarity_search_with_score.return_value = [
        (Document(page_content="Test content 1", metadata={"title": "Doc 1"}), 0.5),
        (Document(page_content="Test content 2", metadata={"title": "Doc 2"}), 0.0),
        (Document(page_content="Test content 3", metadata={"title": "Doc 3"}), -0.5)
    ]
    return vector_store

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Test successful response"))
    return llm

@pytest.fixture
def mock_config():
    config = Mock(spec='CommonConfig')
    
    def get_query_config_side_effect(key=None, default_value=None):
        if key is None:
            return {
                "search": {
                    "rerank_enabled": True,
                    "web_search_enabled": False,
                    "max_retries": 1,
                    "top_k": 10,
                    "relevance_threshold": 0.7
                },
                "hallucination": {
                    "high_risk": 0.6,
                    "medium_risk": 0.8
                },
                "output": {
                    "generate_suggested_documents": True
                },
                "metrics": {
                    "enabled": True,
                    "store_in_db": True,
                    "log_level": "INFO"
                }
            }
        
        keys = key.split(".")
        value = get_query_config_side_effect()
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default_value
    
    config.get_query_config = Mock(side_effect=get_query_config_side_effect)
    config.get_model = Mock(return_value=Mock(invoke=Mock(return_value=Mock(content="mocked response"))))
    return config

@pytest.fixture
def workflow(mock_llm, mock_vector_store, mock_config, mock_logger):
    return QueryProcessWorkflow(mock_llm, mock_vector_store, mock_config)

def test_successful_invoke(workflow, mock_llm):
    # Configure mock for successful response
    mock_llm.invoke.return_value = Mock(content="Test successful response")
    
    result = workflow.invoke(
        user_input="test query",
        user_id="test_user",
        session_id="test_session",
        request_id="test_request"
    )
    
    # Verify successful response
    assert "answer" in result
    assert "citations" in result
    assert "suggested_questions" in result
    assert "metadata" in result
    assert result["answer"] == "Test successful response"
    assert isinstance(result["citations"], list)
    assert isinstance(result["suggested_questions"], list)
    assert isinstance(result["metadata"], dict)
    assert result["metadata"].get("output_format") != ""

def test_error_handling(workflow, mock_llm):
    # Configure mock to return empty response for all LLM calls during error case
    def mock_invoke(*args, **kwargs):
        if isinstance(args[0][0], HumanMessage):
            # Return empty response for all LLM calls
            return Mock(content="")
        return Mock(content="")
    
    mock_llm.invoke.side_effect = mock_invoke
    
    result = workflow.invoke(
        user_input="test query",
        user_id="test_user",
        session_id="test_session",
        request_id="test_request"
    )
    
    # Verify error handling behavior
    assert "answer" in result
    assert result["answer"] == ""  # Empty response on error
    assert "citations" in result
    assert "suggested_questions" in result
    assert "metadata" in result
    assert isinstance(result["citations"], list)
    assert isinstance(result["suggested_questions"], list)
    assert len(result["citations"]) == 0
    assert len(result["suggested_questions"]) == 0
    assert result["metadata"].get("output_format") == "" 