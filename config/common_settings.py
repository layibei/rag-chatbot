import os

import dotenv
import yaml
from langchain_anthropic import ChatAnthropic, AnthropicLLM
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.llms.sparkllm import SparkLLM
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_redis import RedisVectorStore, RedisConfig

from utils.logging_util import logger


class CommonConfig:
    def __init__(self, config_path: str = None):
        self.logger = logger
        dotenv.load_dotenv(dotenv_path=os.getcwd() + '/.env')
        
        if config_path:
            if not os.path.exists(config_path):
                raise ConfigError("Config file not found")
            self.config = self.load_yaml_file(config_path)
        else:
            default_path = os.getcwd() + "/config/app.yaml"
            if not os.path.exists(default_path):
                raise ConfigError("Config file not found")
            self.config = self.load_yaml_file(default_path)
        
        if not self.config:
            raise ConfigError("Invalid configuration")

    def check_config(self, config, path, message):
        """Helper function to check configuration and raise an error if necessary."""
        current = config
        for key in path:
            if key not in current:
                raise ConfigError(message)
            current = current[key]

    def _get_llm_model(self, config):
        self.check_config(config, ["app", "models", "llm", "type"],
                         "LLM model is not found")
        self.logger.info(f"LLM model: {self.config['app']['models']['llm']}")

        if self.config["app"]["models"]["llm"].get("type") == "sparkllm":
            return SparkLLM()

        if self.config["app"]["models"]["llm"].get("type") == "ollama" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            return OllamaLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            return GoogleGenerativeAI(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        if self.config["app"]["models"]["llm"].get("type") == "anthropic" and self.config["app"]["models"][
            "llm"].get("model") is not None:
            return AnthropicLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        raise RuntimeError("Not found the model type");

    def _get_embedding_model(self, config):
        self.check_config(self.config, ["app", "models", "embedding", "type"],
                              "Embedding model is not found")
        self.logger.info(f"Embedding model: {self.config['app']['models']['embedding']}")

        if self.config["app"]["models"]["embedding"].get("type") == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["embedding"].get("model"))

        raise RuntimeError("Not found the embedding model type");

    def _get_chatllm_model(self, config):
        self.check_config(self.config, ["app", "models", "chatllm", "type"],
                              "ChatLLM model is not found")
        self.logger.info(f"ChatLLM model: {self.config['app']['models']['chatllm']}")

        if self.config["app"]["models"]["chatllm"].get("type") == "sparkllm":
            return ChatSparkLLM()

        if self.config["app"]["models"]["chatllm"].get("type") == "ollama" and self.config["app"]["models"][
            "chatllm"].get("model") is not None:
            return ChatOllama(model=self.config["app"]["models"]["chatllm"].get("model"), temperature=0.85, )

        if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            return ChatGoogleGenerativeAI(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            return ChatAnthropic(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        raise RuntimeError("Not found the chatllm model type");

    def _get_rerank_model(self, config):
        self.check_config(self.config, ["app", "models", "rerank", "type"], "rerank model is not found")
        self.logger.info(f"rerank model:{self.config['app']['models']['rerank']}")
        if self.config["app"]["models"]["rerank"].get("type") == "huggingface" and \
                self.config["app"]["models"]["rerank"]["model"] is not None:
            return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["rerank"].get("model"))

    def get_model(self, type):
        if not isinstance(type, str):
            raise TypeError("Model type must be a string")

        if type == "embedding":
            return self._get_embedding_model(self.config);
        elif type == "llm":
            return self._get_llm_model(self.config);

        elif type == "chatllm":
            return self._get_chatllm_model(self.config);
        elif type == "rerank":
            return self._get_rerank_model(self.config);

        else:
            raise ValueError("Invalid model type")

    def get_embedding_config(self):
        self.check_config(self.config, ["app", "embedding"], "app embedding is not found.")
        self.check_config(self.config, ["app", "embedding", "input_path"], "input path in app embedding is not found.")

        return {
            "input_path": self.config["app"]["embedding"].get("input_path"),
            "trunk_size": self.config["app"]["embedding"].get("trunk_size", 1024)
        }

    def get_query_config(self):
        self.check_config(self.config, ["app", "query_agent"], "query agent config is not found")
        query_agent_config = self.config["app"]["query_agent"]

        return {
            "rerank_enabled": query_agent_config.get("rerank_enabled", False),
            "parent_search_enabled": query_agent_config.get("parent_search_enabled", False),
            "web_search_enabled": query_agent_config.get("web_search_enabled", False),
        }
    def get_vector_store(self):
        config = RedisConfig(
            index_name="rag_docs",
            redis_url=os.environ["REDIS_URL"],
            distance_metric="COSINE",  # Options: COSINE, L2, IP
        )
        vector_store = RedisVectorStore(self.get_model("embedding"), config=config)

        return vector_store

    @staticmethod
    def load_yaml_file(file_path: str):
        try:
            with open(file_path, 'r') as file:
                # Use the safe loader to avoid security risks
                data = yaml.safe_load(file)
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except yaml.YAMLError as exc:
            logger.error(f"Error in YAML file: {exc}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None

    def get_db_url(self) -> str:
        """Get database URL from config"""
        return self.config.get("app", {}).get("database", {}).get("url", "sqlite:///:memory:")

    def get_model_config(self, model_type: str):
        """Get configuration for a specific model type"""
        self.check_config(self.config, ["app", "models", model_type], f"{model_type} model config not found")
        return self.config["app"]["models"][model_type]


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass
