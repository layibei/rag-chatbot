import os

import dotenv
import yaml
from langchain_anthropic import ChatAnthropic, AnthropicLLM
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.llms.sparkllm import SparkLLM
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama

from utils.logging_util import logger


class CommonConfig:
    def __init__(self):
        self.logger = logger
        dotenv.load_dotenv(dotenv_path=os.getcwd() + '/.env')
        self.config = self.load_yaml_file(os.getcwd() + "/config/app.yaml")

    def check_config(self, config, path, message):
        """Helper function to check configuration and raise an error if necessary."""
        current = config
        for key in path:
            if key not in current:
                raise ConfigError(message)
            current = current[key]

    def get_model(self, type):
        if not isinstance(type, str):
            raise TypeError("Model type must be a string")

        if type == "embedding":
            self.check_config(self.config, ["app", "models", "embedding", "type"],
                              "Embedding model not configured appropriately in app.yaml")
            self.logger.info(f"Embedding model: {self.config['app']['models']['embedding']}")

            if self.config["app"]["models"]["embedding"].get("type") == "huggingface":
                return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["embedding"].get("model"))

        elif type == "llm":
            self.check_config(self.config, ["app", "models", "llm", "type"],
                              "LLM model not configured appropriately in app.yaml")
            self.logger.info(f"LLM model: {self.config['app']['models']['llm']}")

            if self.config["app"]["models"]["llm"].get("type") == "sparkllm":
                return SparkLLM()

            if self.config["app"]["models"]["llm"].get("type") == "ollama" and self.config["app"]["models"]["llm"].get(
                    "model") != None:
                return OllamaLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

            if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                    "model") != None:
                return GoogleGenerativeAI(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

            if self.config["app"]["models"]["llm"].get("type") == "anthropic" and self.config["app"]["models"]["llm"].get(
                    "model") != None:
                return AnthropicLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        elif type == "chatllm":
            self.check_config(self.config, ["app", "models", "chatllm", "type"],
                              "ChatLLM model not configured appropriately in app.yaml")
            self.logger.info(f"ChatLLM model: {self.config['app']['models']['chatllm']}")

            if self.config["app"]["models"]["chatllm"].get("type") == "sparkllm":
                return ChatSparkLLM()

            if self.config["app"]["models"]["chatllm"].get("type") == "ollama" and self.config["app"]["models"][
                "chatllm"].get("model") != None:
                return ChatOllama(model=self.config["app"]["models"]["chatllm"].get("model"), temperature=0.85, )

            if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                    "model") != None:
                return ChatGoogleGenerativeAI(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

            if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                    "model") != None:
                return ChatAnthropic(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        else:
            raise ValueError("Invalid model type")

    def get_embedding_config(self):
        self.check_config(self.config, ["app", "embedding", "input_path"], "input path cannot be null.")
        self.check_config(self.config, ["app", "embedding", "archive_path"], "archive path cannot be null.")
        #self.check_config(self.config, ["app", "embedding", "trunk_size"], "not define the default trunk size")
        #self.check_config(self.config, ["app", "embedding", "overlap"], "overlap is not found")

        return {
            "input_path": self.config["app"]["embedding"].get("input_path"),
            "archive_path": self.config["app"]["embedding"].get("archive_path"),
            "trunk_size": self.config["app"]["embedding"].get("trunk_size"),
            "overlap": self.config["app"]["embedding"].get("overlap")
        }

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


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass
