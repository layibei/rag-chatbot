import os
import traceback
from typing import Dict, Any

from pydantic import BaseModel

from conversation.conversation_history_helper import ConversationHistoryHelper
from conversation.repositories import ConversationHistoryRepository
from handler.workflow.query_process_workflow import QueryProcessWorkflow, QueryResponse
from utils.logging_util import logger
from utils.prompt_loader import load_txt_prompt

# Get absolute path to the project root
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))


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
        self.conversation_helper = ConversationHistoryHelper(
            ConversationHistoryRepository(self.config.get_db_manager()))

        # Use absolute path for loading prompts
        prompt_path = os.path.join(PROJECT_ROOT, "handler", "prompts", "semantic_route.txt")
        self.semantic_router_prompt = load_txt_prompt(
            prompt_path,
            input_variables=["user_input"]
        )

    def handle(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[str, Any]:
        """Main handler for processing user queries"""
        self.logger.info(f"Handling user query, request_id:{request_id}, user_input:{user_id}")

        # First, route the query
        route = self._route_query(user_input, user_id, session_id, request_id)

        # Process based on routing result, curretly we only support greeting and domain query, in the future we will support more types if any.
        if route == "GREETING":
            return self._process_greeting_query(user_input, user_id, session_id, request_id)
        elif route == "DOMAIN_QUERY":
            return self._process_domain_query(user_input, user_id, session_id, request_id)
        else:  # UNKNOWN
            return self._process_domain_query(user_input, user_id, session_id, request_id)

    def _process_domain_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing user domain query, request_id:{request_id}, user_input:{user_id}")

            # add conversation history to the user input - the top 10 messages
            conversation_history = self.conversation_helper.get_conversation_history(user_id, session_id, limit=10)

            # log the count of histories loaded
            self.logger.info(f"Loaded {len(conversation_history)} conversation histories for user {user_id}")

            if len(conversation_history) > 0:
                conversation_history_str = "\n".join(
                    [f"{msg.user_input} ==> {msg.response}" for msg in conversation_history])
                user_input = f"{user_input}\n\nConversation History:\n{conversation_history_str}"

            # Process query
            response = self._process_query(user_input, user_id, session_id, request_id)

            # Track conversation response
            self.conversation_helper.save_conversation(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                user_input=user_input,
                response=str(response) if response is not None else ""
            )
            self.logger.info(
                f"Query processed successfully, request_id:{request_id}, user_input:{user_id}, response:{response}")
            return response

        except Exception as e:
            self.logger.error(
                f"Error in query processing: {str(e)}\n"
                f"User Input: {user_input}\n"
                f"Request ID: {request_id}\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )
            raise

    def _route_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> str:
        """Route the query to appropriate handler based on semantic content"""
        self.logger.info(f"Routing query, user_input:{user_input}, request_id:{request_id}")

        router_response = self.llm.invoke(
            self.semantic_router_prompt.format(user_input=user_input)
        )

        return router_response.content.strip()

    def _process_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing query, user_id:{user_id}, session_id:{session_id}, request_id:{request_id}, "
                             f"user_input:{user_id}")
            workflow = QueryProcessWorkflow(self.llm, self.vector_store, self.config)
            return workflow.invoke(user_input, user_id=user_id, request_id=request_id, session_id=session_id)
        except Exception as e:
            self.logger.error(
                f"Error in query processing: {str(e)}\n"
                f"User Input: {user_input}\n"
                f"Request ID: {request_id}\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )
            raise

    def _process_greeting_query(self, user_input: str, user_id: str, session_id: str, request_id: str) -> Dict[
        str, Any]:
        """Process greeting messages with a friendly response"""
        self.logger.info(f"Processing greeting query, request_id:{request_id}, user_input:{user_id}")
        
        # Use LLM to generate a single contextual greeting
        greeting_prompt = """Generate a single, friendly greeting response to the user's message. 
        The response should be:
        - Short and natural (1-2 sentences max)
        - Warm and welcoming
        - No multiple options or explanations
        - No markdown formatting

        User message: """ + user_input
        
        self.logger.debug(f"Greeting prompt: {greeting_prompt}")
        response = self.llm.invoke(greeting_prompt).content.strip()
        self.logger.debug(f"Greeting response: {response}")
        
        # Track conversation
        self.conversation_helper.save_conversation(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            user_input=user_input,
            response=response
        )
        self.logger.info(f"Query processed successfully, request_id:{request_id}, user_input:{user_id}, response:{response}")

        return QueryResponse(answer=response, citations=[], suggested_questions=[], metadata={})



if __name__ == "__main__":
    from config.common_settings import CommonConfig

    config = CommonConfig()
    config.setup_proxy()
    handler = QueryHandler(config.get_model("chatllm"), config.get_vector_store(), config)
    handler.handle("Hello!", "user_id", "session_id", "request_id")
