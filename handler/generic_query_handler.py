from typing import Dict, Any
import traceback

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from conversation.repositories import ConversationHistoryRepository
from utils.logging_util import logger
from conversation.conversation_history_helper import ConversationHistoryHelper
from handler.workflow.query_process_workflow import QueryProcessWorkflow
import traceback
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from conversation.conversation_history_helper import ConversationHistoryHelper
from conversation.repositories import ConversationHistoryRepository
from handler.workflow.query_process_workflow import QueryProcessWorkflow
from utils.logging_util import logger


class QueryResponse(BaseModel):
    data: Any
    user_input: str
    request_id: str
    latency_ms: int


class QueryError(BaseModel):
    status: str = "error"
    error_message: str
    user_input: str
    error_code: str


class QueryHandler:
    def __init__(self, llm, vector_store, config):
        self.config = config
        self.llm = llm
        self.vector_store = vector_store
        self.logger = logger
        self.conversation_helper = ConversationHistoryHelper(ConversationHistoryRepository(self.config.get_db_manager()))

    def handle(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Handling user query, user_id:{user_id}, session_id:{session_id}, request_id:{request_id}, user_input:{user_id}")
            # Process query
            if self._is_greeting(user_input, user_id, request_id):
                response = self._handle_greeting()
            else:
                response = self._process_query(user_input, user_id, session_id, request_id)

            # Track conversation response
            self.conversation_helper.save_conversation(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                user_input=user_input,
                response=response
            )

            return {
                "data": response,
                "status": "success",
                "user_input": user_input,
                "session_id": session_id,
                "request_id": request_id
            }

        except Exception as e:
            self.logger.error(
                f"Error processing query: {str(e)}\n"
                f"User Input: {user_input}\n"
                f"User ID: {user_id}\n"
                f"Session ID: {session_id}\n"
                f"Request ID: {request_id}\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )
            return {
                "error": str(e),
                "status": "error",
                "user_input": user_input,
                "session_id": session_id,
                "user_id": user_id,
                "request_id": request_id
            }

    def _process_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> str:
        try:
            self.logger.info(f"Processing query, user_id:{user_id}, session_id:{session_id}, request_id:{request_id}, user_input:{user_id}")
            workflow = QueryProcessWorkflow(self.llm, self.vector_store, self.config)
            return workflow.invoke(user_input, user_id=user_id, request_id=request_id, session_id=session_id)
        except Exception as e:
            self.logger.error(
                f"Error in query processing: {str(e)}\n"
                f"User Input: {user_input}\n"
                f"User ID: {user_id}\n"
                f"Session ID: {session_id}\n"
                f"Request ID: {request_id}\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )
            raise

    def _is_greeting(self, user_input: str, user_id: str, request_id: str) -> bool:
        """Check if input is a greeting using a simple, focused prompt."""
        self.logger.info(f"Checking if input is a greeting, user_id:{user_id}, request_id:{request_id}, user_input:{user_id}")
        # Ensure inputs are valid strings
        if not all(isinstance(x, str) for x in [user_input, user_id, request_id]):
            raise ValueError("All inputs must be strings")

        try:
            # Create a message for the LLM using the newer invoke method
            message = HumanMessage(content=f"Is this a greeting? Respond with 'true' or 'false': {user_input}")
            response = self.llm.invoke([message]).content

            return str(response).lower() == "true"
        except Exception as e:
            self.logger.error(
                f"Error in greeting check: {str(e)}\n"
                f"User Input: {user_input}\n"
                f"User ID: {user_id}\n"
                f"Request ID: {request_id}\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )
            raise  # Re-raise to be handled by the main error handler

    def _handle_greeting(self) -> str:
        self.logger.info("Handling greeting")
        return "Hello! How can I help you today?"

if __name__ == "__main__":
    from config.common_settings import CommonConfig
    config = CommonConfig()
    config.setup_proxy()
    handler = QueryHandler(config.get_model("chatllm"), config.get_vector_store(), config)
    handler.handle("Hello!", "user_id", "session_id", "request_id")
