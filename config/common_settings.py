import os
from pathlib import Path
from typing import Any, Dict, Union
from functools import lru_cache
import dotenv
import yaml
from spacy.language import Language
from utils.logger_init import logger
from utils.async_mdc import setup_mdc

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)


class CommonConfig:
    def __init__(self, config_path: str = None):
        # Initialize MDC handling
        setup_mdc()
        self.logger = logger
        dotenv_path = os.path.join(BASE_DIR, '..', '.env')
        if os.path.exists(dotenv_path):
            dotenv.load_dotenv(dotenv_path)
        elif os.path.exists(os.getenv('DOTENV_PATH')):
            dotenv.load_dotenv(dotenv_path=os.getenv('DOTENV_PATH'))
        else:
            self.logger.warning(f".env file not found at{dotenv_path}")

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

        model_type = self.config["app"]["models"]["llm"].get("type")
        model_name = self.config["app"]["models"]["llm"].get("model")

        if model_type == "sparkllm":
            from langchain_community.chat_models import ChatSparkLLM
            return ChatSparkLLM()

        if model_type == "ollama" and model_name is not None:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=model_name, temperature=0.85)

        if model_type == "gemini" and model_name is not None:
            from langchain_google_genai import GoogleGenerativeAI
            return GoogleGenerativeAI(model=model_name, temperature=0.85)

        if model_type == "anthropic" and model_name is not None:
            from langchain_anthropic import AnthropicLLM
            return AnthropicLLM(model=model_name, temperature=0.85)

        raise RuntimeError("Not found the model type")

    def _get_embedding_model(self):
        self.check_config(self.config, ["app", "models", "embedding", "type"],
                          "Embedding model is not found")
        self.logger.info(f"Embedding model: {self.config['app']['models']['embedding']}")

        if self.config["app"]["models"]["embedding"].get("type") == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["embedding"].get("model"))

        raise RuntimeError("Not found the embedding model type")

    def _get_chatllm_model(self):
        """Get chat LLM model with proxy configuration"""
        try:
            self.check_config(self.config, ["app", "models", "chatllm", "type"],
                              "ChatLLM model is not found")
            model_config = self.config["app"]["models"]["chatllm"]
            model_type = model_config.get("type")
            model_name = model_config.get("model")

            if model_type == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.85,
                )
            elif model_type == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=model_name,
                    temperature=0.85,
                )

            raise RuntimeError(f"Unsupported chatllm model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error initializing chat LLM: {str(e)}")
            raise

    def _get_rerank_model(self):
        self.check_config(self.config, ["app", "models", "rerank", "type"], "rerank model is not found")
        self.logger.info(f"rerank model:{self.config['app']['models']['rerank']}")

        model_type = self.config["app"]["models"]["rerank"].get("type")
        model_name = self.config["app"]["models"]["rerank"].get("model")

        if model_type == "cross-encoder" and model_name is not None:
            from sentence_transformers import CrossEncoder
            model_path = Path(os.path.join(BASE_DIR, "../models/models--cross-encoder--ms-marco-MiniLM-L12-v2"))
            return CrossEncoder(
                # 'cross-encoder/ms-marco-MiniLM-L12-v2',
                model_path,
                max_length=1024,
                local_files_only=True
            )

        if model_type == "bge" and model_name is not None:
            from FlagEmbedding import FlagReranker
            return FlagReranker(model_name, use_fp16=True)

    def get_tokenizer(self):
        """Get tokenizer by model name"""
        model_path = Path(os.path.join(BASE_DIR, "../models/models--cross-encoder--ms-marco-MiniLM-L12-v2"))
        self.logger.info(f"Get tokenizer by model name: {model_path}")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    def get_model(self, type):
        """Get model by type"""
        self.logger.info(f"Get model by type: {type}")
        if not isinstance(type, str):
            raise TypeError("Model type must be a string")

        if type == "embedding":
            return self._get_embedding_model()
        elif type == "llm":
            return self._get_llm_model()
        elif type == "chatllm":
            return self._get_chatllm_model()
        elif type == "rerank":
            return self._get_rerank_model()
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
                "username": os.environ.get("CONFLUENCE_USER_NAME"),
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
                "relevance_threshold": query_agent_config.get("search", {}).get("relevance_threshold", 0.7),
                "query_expansion_enabled": query_agent_config.get("search", {}).get("query_expansion_enabled", False),
                "graph_search_enabled": query_agent_config.get("search", {}).get("graph_search_enabled", False),
                "hypothetical_answer_enabled": query_agent_config.get("search", {}).get("hypothetical_answer_enabled",
                                                                                        False),
            },
            "grading": {
                "minimum_score": query_agent_config.get("grading", {}).get("minimum_score", 0.7),
            },
            "output": {
                "generate_suggested_documents": query_agent_config.get("output", {}).get("generate_suggested_documents",
                                                                                         False),
                "generate_citations": query_agent_config.get("output", {}).get("generate_suggested_documents", False),
                "format": {
                    "default": query_agent_config.get("output", {}).get("format", {}).get("default", "markdown"),
                    "detect_from_query": query_agent_config.get("output", {}).get("format", {}).get("detect_from_query",
                                                                                                    True),
                    "include_metadata": query_agent_config.get("output", {}).get("format", {}).get("include_metadata",
                                                                                                   True),
                }
            },
            "metrics": {
                "enabled": query_agent_config.get("metrics", {}).get("enabled", True),
                "store_in_db": query_agent_config.get("metrics", {}).get("store_in_db", True),
                "log_level": query_agent_config.get("metrics", {}).get("log_level", "INFO")
            },
            "cache": {
                "enabled": query_agent_config.get("cache", {}).get("enabled", False),
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

    @lru_cache(maxsize=1)
    def get_vector_store(self, isForCache=False) -> 'VectorStore':
        """Get vector store"""
        self.logger.info("Get vector store.")
        self.check_config(self.config, ["app", "embedding", "vector_store"], "app vector_store is not found.")
        vector_store_type = self.config["app"]["embedding"]["vector_store"].get("type")
        collection_name = self.config["app"]["embedding"]["vector_store"].get("collection_name")
        if isForCache and self.config["app"]["query_agent"]["cache"]["enabled"]:
            collection_name = self.config["app"]["embedding"]["vector_store"].get("cache_collection_name")

        if vector_store_type == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore.from_documents(
                documents=[],
                embedding=self.get_model("embedding"),
                collection_name=collection_name,
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
            )

        elif vector_store_type == "redis":
            from langchain_redis import RedisConfig, RedisVectorStore
            config = RedisConfig(
                index_name=collection_name,
                redis_url=os.environ["REDIS_URL"],
                distance_metric="COSINE",  # Options: COSINE, L2, IP
            )
            return RedisVectorStore(self.get_model("embedding"), config=config)

        elif vector_store_type == "pgvector":
            from langchain_postgres import PGVector
            return PGVector(
                embeddings=self.get_model("embedding"),
                collection_name=collection_name,
                connection=os.environ["POSTGRES_URI"],
                use_jsonb=True,
            )
        else:
            raise RuntimeError("Not found the vector store type")

    @lru_cache(maxsize=1)
    def get_graph_store(self):
        """Get Neo4j graph store"""
        try:
            if not self.config["app"]["embedding"]["graph_store"].get("enabled", False):
                self.logger.info("Graph store is disabled")
                return None

            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USERNAME")
            password = os.environ.get("NEO4J_PASSWORD")

            if not all([uri, username, password]):
                self.logger.error("Missing Neo4j credentials in environment variables")
                return None

            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(username, password))

            # Test the connection
            try:
                driver.verify_connectivity()
                self.logger.info("Successfully connected to Neo4j")
                return driver
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
                if driver:
                    driver.close()
                return None

        except Exception as e:
            self.logger.error(f"Failed to initialize graph store: {str(e)}")
            return None

    def get_nlp_spacy(self) -> Language:
        """Get NLP model"""
        import spacy

        # Load spaCy model from local path
        model_path = Path(os.path.join(BASE_DIR, "../models/spacy/en_core_web_md"))
        if not model_path.exists():
            raise RuntimeError(
                "spaCy model not found. Please run scripts/download_spacy_model.py first"
            )
        return spacy.load(str(model_path))

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
        """Get database manager instance"""
        from config.database.database_manager import DatabaseManager
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

    def get_env_variable(self, key: str) -> str:
        """Get environment variable"""
        return os.environ.get(key)


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


if __name__ == "__main__":
    config = CommonConfig()
    web_search_enabled = config.get_query_config("search.web_search_enabled")
    print(web_search_enabled)
    print(config.get_logging_config("utils.lock"))
