from typing import List, Optional
from sqlalchemy import func
from datetime import datetime, UTC

from config.database.repository import BaseRepository
from conversation import ConversationHistory, ChatSession
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
            results = session.query(ConversationHistory) \
                .filter_by(
                    user_id=user_id, 
                    session_id=session_id,
                    is_deleted=False
                ) \
                .order_by(ConversationHistory.created_at.desc()) \
                .limit(limit) \
                .all()
            return [self._create_detached_copy(result) for result in results]

    def find_by_user(self, user_id: str) -> List[ConversationHistory]:
        with self.db_manager.session() as session:
            results = session.query(ConversationHistory) \
                .filter_by(user_id=user_id) \
                .order_by(ConversationHistory.created_at.asc()) \
                .limit(1) \
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
            is_deleted=db_obj.is_deleted,
            modified_at=db_obj.modified_at,
            liked=db_obj.liked,
            created_by=db_obj.created_by,
            modified_by=db_obj.modified_by
        )
    
    def get_session_list(self, user_id: str) -> List[ChatSession]:
        with self.db_manager.session() as session:
            # Subquery to get the earliest message for each non-deleted session
            earliest_messages = (
                session.query(
                    ConversationHistory.session_id,
                    func.min(ConversationHistory.created_at).label('first_message_time')
                )
                .filter(
                    ConversationHistory.user_id == user_id,
                    ConversationHistory.is_deleted == False
                )
                .group_by(ConversationHistory.session_id)
                .subquery()
            )
            # Join with the main table to get the user_input for these earliest messages
            results = (
                session.query(
                    ConversationHistory.session_id,
                    ConversationHistory.user_input
                )
                .join(
                    earliest_messages,
                    (ConversationHistory.session_id == earliest_messages.c.session_id) &
                    (ConversationHistory.created_at == earliest_messages.c.first_message_time)
                )
                .filter(
                    ConversationHistory.user_id == user_id,
                    ConversationHistory.is_deleted == False
                )
                .all()
            )
            return [
                ChatSession(
                    session_id=result.session_id,
                    title=result.user_input
                )
                for result in results
            ]

    def update_message_like(self, 
                           user_id: str, 
                           session_id: str, 
                           request_id: str,
                           liked: bool) -> Optional[ConversationHistory]:
        """Update the liked status of a message and return the updated message"""
        with self.db_manager.session() as session:
            message = session.query(ConversationHistory).filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.session_id == session_id,
                ConversationHistory.request_id == request_id,
                ConversationHistory.is_deleted == False
            ).first()
            
            if not message:
                return None
            
            message.liked = liked
            message.modified_at = datetime.now(UTC)
            session.commit()
            return self._create_detached_copy(message)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Mark all messages in a session as deleted"""
        with self.db_manager.session() as session:
            result = session.query(ConversationHistory).filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.session_id == session_id,
                ConversationHistory.is_deleted == False
            ).update({
                ConversationHistory.is_deleted: True,
                ConversationHistory.modified_at: datetime.now(UTC)
            })
            session.commit()
            return result > 0