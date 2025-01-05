import os
from typing import Any, Dict, Union
from functools import lru_cache

import dotenv
import yaml
from langchain_anthropic import ChatAnthropic, AnthropicLLM
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.llms.sparkllm import SparkLLM
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_postgres import PGVector
from FlagEmbedding import FlagReranker

from config.database.database_manager import DatabaseManager
from utils.logger_init import logger

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)


class CommonConfig:
    def __init__(self, config_path: str = None):
        self.logger = logger
        dotenv.load_dotenv(dotenv_path=BASE_DIR + '/../.env')

        if config_path:
            path = BASE_DIR + config_path
            if not os.path.exists(path):
                raise ConfigError("Config file not found")
            self.config = self.load_yaml_file(path)
        else:
            default_path = BASE_DIR + "/app.yaml"
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
        """Get chat LLM model with proxy configuration"""
        try:
            self.check_config(self.config, ["app", "models", "chatllm", "type"],
                              "ChatLLM model is not found")
            model_config = self.config["app"]["models"]["chatllm"]
            model_type = model_config.get("type")

            if model_type == "gemini":
                return ChatGoogleGenerativeAI(
                    model=model_config.get("model"),
                    temperature=0.85,
                )
            elif model_type == "anthropic":
                return ChatAnthropic(
                    model=model_config.get("model"),
                    temperature=0.85,
                )
            # Add other model types as needed...

            raise RuntimeError(f"Unsupported chatllm model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error initializing chat LLM: {str(e)}")
            raise

    def _get_rerank_model(self, config):
        self.check_config(self.config, ["app", "models", "rerank", "type"], "rerank model is not found")
        self.logger.info(f"rerank model:{self.config['app']['models']['rerank']}")
        if self.config["app"]["models"]["rerank"].get("type") == "huggingface" and \
                self.config["app"]["models"]["rerank"]["model"] is not None:
            return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["rerank"].get("model"))
        if self.config["app"]["models"]["rerank"].get("type") == "bge" and \
                self.config["app"]["models"]["rerank"]["model"] is not None:
            return FlagReranker(self.config["app"]["models"]["rerank"].get("model"), use_fp16=True)

    def get_model(self, type):
        """Get model by type"""
        self.logger.info(f"Get model by type: {type}")
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

    def get_embedding_config(self, key: str = None, default_value: Any = None) -> Any:
        self.logger.info(f"Embedding config: {self.config['app']['embedding']}")
        self.check_config(self.config, ["app", "embedding"], "app embedding is not found.")
        self.check_config(self.config, ["app", "embedding", "input_path"], "input path in app embedding is not found.")
        embedding_config = {
            "input_path": self.config["app"]["embedding"].get("input_path"),
            "staging_path": self.config["app"]["embedding"].get("staging_path"),
            "archive_path": self.config["app"]["embedding"].get("archive_path"),
            "trunk_size": self.config["app"]["embedding"].get("trunk_size", 1024),
            "overlap": self.config["app"]["embedding"].get("overlap", 100),
            "confluence": {
                "url": self.config["app"]["embedding"].get("confluence", {}).get("url"),
                "username": os.environ.get("CONFLUENCE_USERNAME"),
                "api_key": os.environ.get("CONFLUENCE_API_KEY"),
            }
        }

        if key is None:
            return embedding_config

        # Handle nested key access
        keys = key.split(".")
        value = embedding_config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default_value

    @lru_cache(maxsize=128)
    def get_query_config(self, key: str = None, default_value: Any = None) -> Any:
        """
        Get query agent configuration with nested key support and caching.
        Example: get_query_config("search.rerank_enabled", True)
        """
        self.logger.info(f"Getting query config for key: {key}")
        self.check_config(self.config, ["app", "query_agent"], "Query agent config is not found")
        query_agent_config = self.config["app"]["query_agent"]

        query_config = {
            "search": {
                "rerank_enabled": query_agent_config.get("search", {}).get("rerank_enabled", False),
                "web_search_enabled": query_agent_config.get("search", {}).get("web_search_enabled", False),
                "max_retries": query_agent_config.get("search", {}).get("max_retries", 1),
                "top_k": query_agent_config.get("search", {}).get("top_k", 10),
                "relevance_threshold": query_agent_config.get("search", {}).get("relevance_threshold", 0.7)
            },
            "grading": {
                "minimum_score": query_agent_config.get("grading", {}).get("minimum_score", 0.7),
            },
            "output": {
                "generate_suggested_documents": query_agent_config.get("output", {}).get("generate_suggested_documents",
                                                                                         False),
                "generate_citations": query_agent_config.get("output", {}).get("generate_suggested_documents", False)
            },
            "metrics": {
                "enabled": query_agent_config.get("metrics", {}).get("enabled", True),
                "store_in_db": query_agent_config.get("metrics", {}).get("store_in_db", True),
                "log_level": query_agent_config.get("metrics", {}).get("log_level", "INFO")
            }
        }

        if key is None:
            return query_config

        # Handle nested key access
        keys = key.split(".")
        value = query_agent_config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default_value

    def get_vector_store(self):
        self.logger.info("Get vector store.")
        # config = RedisConfig(
        #     index_name="rag_docs",
        #     redis_url=os.environ["REDIS_URL"],
        #     distance_metric="COSINE",  # Options: COSINE, L2, IP
        # )
        # vector_store = RedisVectorStore(self.get_model("embedding"), config=config)

        vector_store = PGVector(
            embeddings=self.get_model("embedding"),
            collection_name="rag_docs",
            connection=os.environ["POSTGRES_URI"],
            use_jsonb=True,
        )

        return vector_store

    def setup_proxy(self):
        """Setup proxy configuration"""
        try:
            if self.config["app"]["proxy"].get("enabled", False):
                # Set proxy environment variables
                proxy_config = {
                    "http_proxy": self.config["app"]["proxy"].get("http_proxy"),
                    "https_proxy": self.config["app"]["proxy"].get("https_proxy"),
                    "no_proxy": self.config["app"]["proxy"].get("no_proxy")
                }

                # Set environment variables
                for key, value in proxy_config.items():
                    if value:
                        os.environ[key] = value
                        os.environ[key.upper()] = value  # Some libraries use uppercase

                self.logger.info("Proxy configuration set successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup proxy: {str(e)}")
            raise ConfigError(f"Proxy setup failed: {str(e)}")

    async def asetup_proxy(self):
        """Setup proxy configuration"""
        try:
            if self.config["app"]["proxy"].get("enabled", False):
                # Set proxy environment variables
                proxy_config = {
                    "http_proxy": self.config["app"]["proxy"].get("http_proxy"),
                    "https_proxy": self.config["app"]["proxy"].get("https_proxy"),
                    "no_proxy": self.config["app"]["proxy"].get("no_proxy")
                }

                # Set environment variables
                for key, value in proxy_config.items():
                    if value:
                        os.environ[key] = value
                        os.environ[key.upper()] = value  # Some libraries use uppercase

                self.logger.info("Proxy configuration set successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup proxy: {str(e)}")
            raise ConfigError(f"Proxy setup failed: {str(e)}")

    def get_db_manager(self):
        return DatabaseManager(os.environ["POSTGRES_URI"])

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

    def get_logging_config(self, package_name: str = None) -> Union[Dict[str, str], str]:
        """
        Get logging configuration for packages with hierarchical path support.
        Args:
            package_name: Optional package name to get specific log level
        Returns:
            Dict of package log levels or specific level string
        """
        self.logger.debug(f"Getting logging config for package: {package_name}")

        try:
            # Get logging config with default fallback
            logging_levels = self.config.get("app", {}).get("logging.level", {})
            root_level = logging_levels.get("root", "INFO")

            if package_name:
                # Find the most specific matching package path
                matching_level = root_level
                matching_length = 0

                for pkg_path, level in logging_levels.items():
                    if pkg_path != "root" and package_name.startswith(pkg_path):
                        path_length = len(pkg_path.split('.'))
                        if path_length > matching_length:
                            matching_level = level
                            matching_length = path_length

                return matching_level

            return logging_levels
        except Exception as e:
            self.logger.error(f"Error getting logging config: {str(e)}")
            return "INFO" if package_name else {"root": "INFO"}


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


if __name__ == "__main__":
    config = CommonConfig()
    # config.setup_proxy()
    # llm = config.get_model("chatllm")
    # logger.info(llm.invoke("What is the capital of France?"))

    web_search_enabled = config.get_query_config("search.web_search_enabled")
    print(web_search_enabled)

    print(config.get_logging_config("utils.lock"))
