from langchain_core.language_models import BaseChatModel
from utils.logging_util import logger


class QueryHandler:
    def __init__(self, llm: BaseChatModel):
        self.logger = logger
        self.llm = llm

    def handle(self, user_input: str):


