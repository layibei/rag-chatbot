from datetime import datetime, UTC

from conversation import ConversationHistory
from conversation.repositories import ConversationHistoryRepository


class ConversationHistoryHelper:
    def __init__(self, repository: ConversationHistoryRepository):
        self.repository = repository
        
    def save_conversation(self, 
                         user_id: str, 
                         session_id: str,
                         request_id: str,
                         user_input: str,
                         response: str) -> ConversationHistory:
        conversation = ConversationHistory(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            user_input=user_input,
            response=response,
            created_at=datetime.now(UTC)
        )
        return self.repository.create(conversation)
        
    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 5):
        return self.repository.find_by_session(user_id, session_id, limit) 