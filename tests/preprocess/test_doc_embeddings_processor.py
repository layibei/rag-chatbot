import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC
from preprocess.doc_index_log_processor import DocEmbeddingsProcessor
from preprocess.index_log import Status, IndexLog

@pytest.fixture
def mock_components():
    return {
        'embeddings': Mock(),
        'vector_store': Mock(),
        'index_log_helper': Mock()
    }

@pytest.fixture
def mock_config():
    with patch('config.common_settings.CommonConfig') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.get_db_url.return_value = "sqlite:///:memory:"
        mock_instance.config = {
            'app': {
                'database': {'url': 'sqlite:///:memory:'},
                'embedding': {
                    'input_path': '/test/input',
                    'trunk_size': 512
                }
            }
        }
        return mock_class

@pytest.fixture
def processor(mock_components, mock_config):
    with patch('preprocess.doc_embeddings_processor.CommonConfig', mock_config):
        return DocEmbeddingsProcessor(**mock_components)

class TestDocEmbeddingsProcessor:
    def test_add_new_document(self, processor, mock_components):
        # Setup
        mock_components['index_log_helper'].find_by_checksum.return_value = None
        mock_components['index_log_helper'].find_by_source.return_value = None
        processor._calculate_checksum = Mock(return_value="test_checksum")

        # Execute
        result = processor.add_index_log(
            source="test.pdf",
            source_type="pdf",
            user_id="test_user"
        )

        # Assert
        assert result["message"] == "Document queued for processing"
        assert result["source"] == "test.pdf"
        assert result["source_type"] == "pdf"

    def test_add_duplicate_document(self, processor, mock_components):
        # Setup
        existing_log = IndexLog(
            id=1,
            source="test.pdf",
            source_type="pdf",
            checksum="test_checksum",
            status=Status.COMPLETED
        )
        mock_components['index_log_helper'].find_by_checksum.return_value = existing_log
        processor._calculate_checksum = Mock(return_value="test_checksum")

        # Execute
        result = processor.add_index_log(
            source="test.pdf",
            source_type="pdf",
            user_id="test_user"
        )

        # Assert
        assert "already exists" in result["message"]
        assert result["id"] == 1

    def test_update_existing_document(self, processor, mock_components):
        # Setup
        existing_log = IndexLog(
            id=1,
            source="test.pdf",
            source_type="pdf",
            checksum="old_checksum",
            status=Status.COMPLETED
        )
        mock_components['index_log_helper'].find_by_checksum.return_value = None
        mock_components['index_log_helper'].find_by_source.return_value = existing_log
        processor._calculate_checksum = Mock(return_value="new_checksum")

        # Execute
        result = processor.add_index_log(
            source="test.pdf",
            source_type="pdf",
            user_id="test_user"
        )

        # Assert
        assert "updated" in result["message"]
        assert result["id"] == 1