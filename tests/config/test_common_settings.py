from unittest.mock import patch, mock_open
from pathlib import Path

import pytest
from langchain_community.llms.sparkllm import SparkLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from config.common_settings import CommonConfig, ConfigError

# Sample YAML configuration for testing
SAMPLE_CONFIG = """
app:
  database:
    url: "sqlite:///:memory:"
  models:
    llm:
      type: "sparkllm"
    chatllm:
      type: "ollama"
      model: "llama2"
    embedding:
      type: "huggingface"
      model: "sentence-transformers/all-mpnet-base-v2"
  embedding:
    input_path: "/input"
    trunk_size: 512
  query_agent:
    rerank_enabled: true
    web_search_enabled: false
"""


@pytest.fixture
def common_config():
    with patch('builtins.open', mock_open(read_data=SAMPLE_CONFIG)):
        with patch('os.path.exists', return_value=True):
            return CommonConfig()


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
    assert config["trunk_size"] == 512


def test_get_query_config(common_config):
    config = common_config.get_query_config()
    assert config["rerank_enabled"] is True
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


def test_config_initialization_with_valid_file(tmp_path):
    config_file = tmp_path / "app.yaml"
    config_file.write_text(SAMPLE_CONFIG)
    config = CommonConfig(str(config_file))
    assert config is not None


def test_config_initialization_with_missing_file():
    with pytest.raises(ConfigError, match="Config file not found"):
        CommonConfig("/nonexistent/path")


def test_config_validation_with_invalid_config(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("""
app:
  models: {}
    """)
    with pytest.raises(ConfigError, match="LLM model is not found"):
        config = CommonConfig(str(config_file))
        config.get_model("llm")  # This should raise the error


def test_get_model_config(common_config):
    model_config = common_config.get_model_config("llm")
    assert model_config["type"] == "sparkllm"
