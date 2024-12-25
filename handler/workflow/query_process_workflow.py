from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph.state import StateGraph

from config.common_settings import CommonConfig
from handler.workflow import RequestState
from handler.workflow.nodes import ProcessNodes
from utils.logging_util import logger


class QueryProcessWorkflow():
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        self.logger = logger
        self.llm = llm
        self.nodes = ProcessNodes(llm, vectorstore, config)
        self.graph = self.__setup_graph()

    def __setup_graph(self):
        # Initialize a default checkpointer if none is provided
        memory = MemorySaver()
        workflow = StateGraph(RequestState)

        workflow.add_node("web_search", self.nodes.web_search)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)

        workflow.set_conditional_entry_point(self.nodes.route_query, {
            "websearch": "web_search",
            "vectorstore": "retrieve_documents",
        })

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve_documents", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges("generate", self.nodes.grade_generation, {
            "successfully": END,
            "failed": END,
        }, )

        return workflow.compile(checkpointer=memory)

    def invoke(self, user_input: str, user_id: str, session_id: str, request_id: str):
        self.logger.info(f"Processing query: {user_input}")
        thread = {
            'configurable': {'thread_id': 1}
        }

        for s in self.graph.stream({
            'user_input': user_input,
            'user_id': user_id,
            'session_id': session_id,
            'request_id': request_id,
        }, thread):
            self.logger.info(s)

        state = self.graph.get_state(thread)
        self.logger.info(f"response:{state.values}")
        return state.values.get("response")
