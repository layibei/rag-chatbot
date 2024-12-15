from unittest.mock import patch, mock_open

import pytest
from langchain_community.llms.sparkllm import SparkLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from config.common_settings import CommonConfig, ConfigError

# Sample YAML configuration for testing
SAMPLE_CONFIG = """
app:
  models:
    llm:
      type: "sparkllm"
    chatllm:
      type: "ollama"
      model: "llama2"
    embedding:
      type: "huggingface"
      model: "sentence-transformers/all-mpnet-base-v2"
    rerank:
      type: "huggingface"
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  embedding:
    input_path: "/input"
    archive_path: "/archive"
    trunk_size: 512
    overlap: 50
  query_agent:
    rerank_enabled: true
    parent_search_enabled: true
    web_search_enabled: false
"""


@pytest.fixture
def common_config():
    with patch('builtins.open', mock_open(read_data=SAMPLE_CONFIG)):
        with patch('os.getcwd', return_value='/fake/path'):
            with patch('dotenv.load_dotenv') as mock_load_dotenv:
                config = CommonConfig()
                return config


def test_init(common_config):
    assert common_config.config is not None
    assert isinstance(common_config.config, dict)


def test_check_config_valid_path(common_config):
    # Test valid configuration path
    common_config.check_config(
        common_config.config,
        ["app", "models", "llm", "type"],
        "Test message"
    )


def test_check_config_invalid_path(common_config):
    # Test invalid configuration path
    with pytest.raises(ConfigError):
        common_config.check_config(
            common_config.config,
            ["app", "invalid", "path"],
            "Invalid path"
        )


def test_get_embedding_config(common_config):
    config = common_config.get_embedding_config()
    assert config["input_path"] == "/input"
    assert config["archive_path"] == "/archive"
    assert config["trunk_size"] == 512
    assert config["overlap"] == 50


def test_get_query_config(common_config):
    config = common_config.get_query_config()
    assert config["rerank_enabled"] is True
    assert config["parent_search_enabled"] is True
    assert config["web_search_enabled"] is False


@pytest.mark.parametrize("model_type,model_class", [
    ("embedding", HuggingFaceEmbeddings),
    ("llm", SparkLLM),
    ("chatllm", ChatOllama),
])
def test_get_model(common_config, model_type, model_class):
    with patch.object(model_class, '__new__', return_value="mocked_model") as mock_new:
        model = common_config.get_model(model_type)
        print(f"Model returned: {model}")
        assert mock_new.called


def test_get_model_invalid_type(common_config):
    with pytest.raises(ValueError, match="Invalid model type"):
        common_config.get_model("invalid_type")


def test_get_model_type_not_string(common_config):
    with pytest.raises(TypeError, match="Model type must be a string"):
        common_config.get_model(123)
