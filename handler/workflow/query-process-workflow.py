from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import StateGraph

from handler.workflow import RequestState
from utils.logging_util import logger


class QueryProcessWorkflow:
    def __init__(self, llm: BaseChatModel):
        self.logger = logger
        self.llm = llm
        self.graph = self.__setup_graph()

    def __setup_graph(self):
        workflow = StateGraph(RequestState)

