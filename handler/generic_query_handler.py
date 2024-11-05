from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from handler.workflow.query_process_workflow import QueryProcessWorkflow
from utils.logging_util import logger


class QueryHandler:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.logger = logger
        self.llm = llm
        self.vectorstore = vectorstore

    def handle(self, user_input: str):
        return QueryProcessWorkflow(self.llm, self.vectorstore).run(user_input)

