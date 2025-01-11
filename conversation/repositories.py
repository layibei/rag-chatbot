from typing import List, Optional, Dict, Any
from sqlalchemy import func
from datetime import datetime, UTC
from sqlalchemy.sql import text

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
                .order_by(ConversationHistory.created_at.asc()) \
                .limit(limit) \
                .all()
            return [self._create_detached_copy(result) for result in results]

    def find_by_user(self, user_id: str) -> List[ConversationHistory]:
        with self.db_manager.session() as session:
            results = session.query(ConversationHistory) \
                .filter_by(user_id=user_id) \
                .order_by(ConversationHistory.created_at.desc()) \
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
        """Get list of chat sessions for a user, with first message as title, sorted by modified_at desc"""
        try:
            with self.db_manager.session() as session:
                query = text("""
                    WITH first_messages AS (
                        SELECT 
                            session_id,
                            user_input as first_message,
                            created_at,
                            ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at ASC) as rn
                        FROM conversation_history
                        WHERE user_id = :user_id AND is_deleted = false
                    ),
                    latest_modified AS (
                        SELECT 
                            session_id,
                            MAX(modified_at) as latest_modified_at
                        FROM conversation_history
                        WHERE user_id = :user_id AND is_deleted = false
                        GROUP BY session_id
                    )
                    SELECT 
                        fm.session_id,
                        fm.first_message as title,
                        lm.latest_modified_at
                    FROM first_messages fm
                    JOIN latest_modified lm ON fm.session_id = lm.session_id
                    WHERE fm.rn = 1
                    ORDER BY lm.latest_modified_at DESC
                """)
                
                result = session.execute(query, {"user_id": user_id})
                
                return [
                    ChatSession(
                        session_id=row.session_id,
                        title=row.title
                    )
                    for row in result
                ]

        except Exception as e:
            self.logger.error(f"Error getting session list: {str(e)}")
            raise

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