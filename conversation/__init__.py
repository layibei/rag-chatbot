from sqlalchemy import Column, String, DateTime, Boolean, Text, BigInteger
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel

Base = declarative_base()

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False)
    session_id = Column(String(128), nullable=False)
    request_id = Column(String(128), nullable=False)
    user_input = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    liked = Column(Boolean, nullable=True)
    created_at = Column(DateTime, nullable=False)
    modified_at = Column(DateTime, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_by = Column(String(128), nullable=False)
    modified_by = Column(String(128), nullable=False)


class ChatSession(BaseModel):
    session_id: str
    title: str