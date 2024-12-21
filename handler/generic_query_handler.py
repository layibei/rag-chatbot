from typing import Dict, Any

from pydantic import BaseModel

from conversation.conversation_history_helper import ConversationHistoryHelper
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
    def __init__(self, llm, vector_store, conversation_repo, config):
        self.config = config
        self.llm = llm
        self.conversation_helper = ConversationHistoryHelper(conversation_repo)
        self.vector_store = vector_store

    def handle(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[str, Any]:
        try:
            # Track conversation input
            self.conversation_helper.save_conversation(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                user_input=user_input,
                response=""  # Will be updated after processing
            )

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
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "status": "error",
                "user_input": user_input,
                "request_id": request_id
            }

    def _process_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> str:
        workflow = QueryProcessWorkflow(self.llm, self.vector_store)
        return workflow.invoke(user_input, user_id=user_id, request_id=request_id, session_id=session_id)

    def _is_greeting(self, user_input: str, user_id: str, request_id: str) -> bool:
        """Check if input is a greeting using a simple, focused prompt."""
        # Ensure inputs are valid strings
        if not all(isinstance(x, str) for x in [user_input, user_id, request_id]):
            raise ValueError("All inputs must be strings")

        try:
            # Use f-string for better readability and security
            prompt = f"Is this a greeting? Respond with 'true' or 'false': {user_input}"
            response = self.llm.predict(prompt)

            return str(response).lower() == "true"
        except Exception as e:
            logger.error(f"Error in greeting check: {str(e)}")
            raise  # Re-raise to be handled by the main error handler

    def _handle_greeting(self) -> str:
        return "Hello! How can I help you today?"
