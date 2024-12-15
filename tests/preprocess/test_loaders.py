import os
import pytest
from unittest.mock import Mock, patch
from preprocess.loader.loader_factories import DocumentLoaderFactory
from preprocess.loader.pdf_loader import PDFDocLoader
from preprocess.loader.json_loader import JsonDocLoader
from preprocess.loader.csv_loader import CSVDocLoader
from preprocess.loader.text_loader import TextDocLoader

@pytest.fixture
def sample_files(tmp_path):
    # Create temporary test files
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"PDF content")
    
    json_file = tmp_path / "test.json"
    json_file.write_text('{"test": "data"}')
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\nval1,val2")
    
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Sample text")
    
    return {
        'pdf': str(pdf_file),
        'json': str(json_file),
        'csv': str(csv_file),
        'txt': str(txt_file)
    }

def test_loader_factory_get_loader():
    # Mock the base config
    with patch('preprocess.loader.loader_factories.DocumentLoaderFactory.loader_mapping', {
        '.pdf': PDFDocLoader,
        '.json': JsonDocLoader,
        '.csv': CSVDocLoader,
        '.txt': TextDocLoader,
    }):
        # Test loader factory returns correct loader types
        assert isinstance(DocumentLoaderFactory.get_loader('test.pdf'), PDFDocLoader)
        assert isinstance(DocumentLoaderFactory.get_loader('test.json'), JsonDocLoader)
        assert isinstance(DocumentLoaderFactory.get_loader('test.csv'), CSVDocLoader)
        assert isinstance(DocumentLoaderFactory.get_loader('test.txt'), TextDocLoader)

def test_loader_factory_invalid_path():
    with patch('preprocess.loader.loader_factories.DocumentLoaderFactory.loader_mapping'):
        with pytest.raises(ValueError, match="Invalid file path"):
            DocumentLoaderFactory.get_loader(None)
        with pytest.raises(ValueError, match="Invalid file path"):
            DocumentLoaderFactory.get_loader('')

@pytest.mark.parametrize("loader_class,file_ext", [
    (PDFDocLoader, '.pdf'),
    (JsonDocLoader, '.json'),
    (CSVDocLoader, '.csv'),
    (TextDocLoader, '.txt'),
])
def test_loader_supported_extensions(loader_class, file_ext):
    with patch('preprocess.loader.base_loader.CommonConfig'):
        loader = loader_class()
        assert loader.is_supported_file_extension(f'test{file_ext}')
        # TextDocLoader is special case as it accepts all extensions
        if loader_class != TextDocLoader:
            assert not loader.is_supported_file_extension('test.unsupported') 