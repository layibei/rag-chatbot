from datetime import datetime, UTC
from typing import Dict, Any

from pydantic import BaseModel

from conversation import ConversationHistory
from conversation.conversation_history_helper import ConversationHistoryHelper
from audit.llm_tracking_helper import LLMTrackingHelper
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
    def __init__(self, config, llm, conversation_repo, llm_interaction_repo, vector_store):
        self.config = config
        self.llm = llm
        self.conversation_helper = ConversationHistoryHelper(conversation_repo)
        self.llm_tracking = LLMTrackingHelper(llm_interaction_repo)
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
                response = self._process_query(user_input, user_id, request_id)

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

    def _process_query(self, user_input: str, user_id: str, request_id: str) -> str:
        workflow = QueryProcessWorkflow(self.llm, self.vector_store)
        return workflow.invoke(user_input, user_id=user_id, request_id=request_id)

    def _is_greeting(self, user_input: str, user_id: str, request_id: str) -> bool:
        """Check if input is a greeting using a simple, focused prompt."""
        result = self.llm_tracking.track_interaction(
            prompt="Classify as greeting (true) or not (false): '{text}'",
            response=self.llm.predict("Classify as greeting (true) or not (false): '{text}'".format(text=user_input)),
            user_id=user_id,
            request_id=request_id
        )
        return result.lower() == "true"

    def _handle_greeting(self) -> str:
        return "Hello! How can I help you today?"

