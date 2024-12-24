from datetime import datetime, UTC
from typing import List, Optional

from conversation import ConversationHistory, ChatSession
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
        return self.repository.save(conversation)
        
    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 5):
        return self.repository.find_by_session(user_id, session_id, limit) 
    


    # Add this method to the existing ConversationHistoryHelper class
    def get_session_list(self, user_id: str) -> List[ChatSession]:
        return self.repository.get_session_list(user_id)

    def update_message_like(self, 
                           user_id: str, 
                           session_id: str, 
                           request_id: str,
                           liked: bool) -> Optional[ConversationHistory]:
        """Update the liked status of a specific message"""
        try:
            return self.repository.update_message_like(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                liked=liked
            )
        except Exception as e:
            self.logger.error(f"Error updating message like status: {str(e)}")
            raise