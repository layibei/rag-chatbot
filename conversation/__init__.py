from sqlalchemy import Column, String, DateTime, Boolean, Text, BigInteger
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ConversationHistory(Base):
    __tablename__ = 'conversation_history'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=False)
    session_id = Column(String(128), nullable=False)
    request_id = Column(String(128), nullable=False)
    user_input = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    liked = Column(Boolean, nullable=True)
    token_usage = Column(Text, nullable=True)