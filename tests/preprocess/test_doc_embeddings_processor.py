import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC
from preprocess.doc_embeddings_processor import DocEmbeddingsProcessor
from preprocess import Status

@pytest.fixture
def mock_components():
    embeddings = Mock()
    vector_store = Mock()
    index_log_helper = Mock()
    return embeddings, vector_store, index_log_helper

@pytest.fixture
def processor(mock_components):
    embeddings, vector_store, index_log_helper = mock_components
    with patch('socket.gethostname'):
        return DocEmbeddingsProcessor(embeddings, vector_store, index_log_helper)

def test_add_index_log_new_document(processor):
    # Mock dependencies
    processor.index_log_helper.find_by_checksum.return_value = None
    processor.index_log_helper.find_by_source.return_value = None
    processor._calculate_checksum = Mock(return_value="test_checksum")
    
    # Test adding new document
    result = processor.add_index_log(
        source="test.pdf",
        source_type="pdf",
        user_id="test_user"
    )
    
    assert result["message"] == "Document queued for processing"
    assert "id" in result
    assert result["source"] == "test.pdf"
    assert result["source_type"] == "pdf"

def test_process_pending_documents(processor):
    # Mock index log
    mock_log = Mock()
    mock_log.source = "test.pdf"
    mock_log.source_type = "pdf"
    
    # Mock dependencies
    processor.index_log_helper.get_next_pending_with_lock.return_value = mock_log
    processor.distributed_lock = Mock()
    processor.distributed_lock.acquire_lock.return_value = True
    # Mock _process_document to avoid actual processing
    processor._process_document = Mock()
    
    # Test processing
    processor.process_pending_documents()
    
    # Verify status changes
    assert hasattr(mock_log, 'status')
    assert mock_log.status == Status.COMPLETED
    processor.index_log_helper.save.assert_called()

def test_process_document(processor):
    # Mock document loader
    mock_loader = Mock()
    mock_loader.load.return_value = [Mock()]
    
    with patch('preprocess.loader.loader_factories.DocumentLoaderFactory.get_loader', 
              return_value=mock_loader):
        mock_log = Mock()
        mock_log.source = "test.pdf"
        mock_log.source_type = "pdf"
        
        processor._process_document(mock_log)
        
        processor.vector_store.add_documents.assert_called_once() 