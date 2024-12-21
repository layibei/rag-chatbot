from typing import List, Optional

from config.database.repository import BaseRepository
from conversation import ConversationHistory
from utils.id_util import get_id

class ConversationHistoryRepository(BaseRepository[ConversationHistory]):
    def __init__(self, db_manager):
        super().__init__(db_manager, ConversationHistory)

    def _get_model_class(self) -> type:
        return ConversationHistory

    def create(self, conversation: ConversationHistory) -> ConversationHistory:
        with self.db_manager.session() as session:
            # Set ID if not provided
            if not conversation.id:
                conversation.id = get_id()
            session.add(conversation)
            session.flush()
            session.refresh(conversation)
            return self._create_detached_copy(conversation)

    def find_by_session(self, user_id: str, session_id: str, limit: int = 5) -> List[ConversationHistory]:
        with self.db_manager.session() as session:
            results = session.query(ConversationHistory)\
                .filter_by(user_id=user_id, session_id=session_id)\
                .order_by(ConversationHistory.created_at.desc())\
                .limit(limit)\
                .all()
            return [self._create_detached_copy(result) for result in results]

    def _create_detached_copy(self, db_obj: Optional[ConversationHistory]) -> Optional[ConversationHistory]:
        if not db_obj:
            return None
            
        return ConversationHistory(
            id=db_obj.id,
            user_id=db_obj.user_id,
            session_id=db_obj.session_id,
            request_id=db_obj.request_id,
            user_input=db_obj.user_input,
            response=db_obj.response,
            created_at=db_obj.created_at,
            liked=db_obj.liked,
            token_usage=db_obj.token_usage
        ) 