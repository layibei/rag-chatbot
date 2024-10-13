import os

import dotenv
import yaml
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.llms.sparkllm import SparkLLM
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logging_util import logger


class CommonConfig:
    def __init__(self):
        self.logger = logger
        dotenv.load_dotenv(dotenv_path=os.getcwd() + '/.env')
        self.config = self.load_yaml_file(os.getcwd() + "/config/app.yaml")

    def get_model(self, type: str):
        """
            app:
              embedding:
                type: "huggingface"
                model: "BAAI/bge-large-en-v1.5"
              llm:
            #    type: "huggingface"
            #    model: "meta-llama/Llama-2-13b-chat-hf"
                type: "sparkllm"
              chatllm:
                type: "sparkllm
        """
        # Retrieve the model configuration based on the type
        if type == "embedding":
            if "embedding" not in self.config["app"] or "type" not in self.config["app"]["embedding"] and "model" not in \
                    self.config["app"]["embedding"]:
                raise ValueError("Embedding model not configured appropriately in app.yaml")
            self.logger.info(f"Embedding model: {self.config['app']['embedding']}")

            if self.config["app"]["embedding"]["type"] == "huggingface":
                return HuggingFaceEmbeddings(model_name=self.config["app"]["embedding"]["model"])
        elif type == "llm":
            if "llm" not in self.config["app"] or "type" not in self.config["app"]["llm"]:
                raise ValueError("LLM model not configured appropriately in app.yaml")
            self.logger.info(f"LLM model: {self.config['app']['llm']}")

            if self.config["app"]["llm"]["type"] == "sparkllm":
                return SparkLLM()
        elif type == "chatllm":
            if "chatllm" not in self.config["app"] or "type" not in self.config["app"]["chatllm"]:
                raise ValueError("ChatLLM model not configured appropriately in app.yaml")
            self.logger.info(f"ChatLLM model: {self.config['app']['chatllm']}")

            if self.config["app"]["chatllm"]["type"] == "sparkllm":
                return ChatSparkLLM()
        else:
            raise ValueError("Invalid model type")

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
