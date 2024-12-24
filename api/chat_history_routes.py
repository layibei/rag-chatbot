from pydantic import BaseModel
from typing import List
from conversation import ChatSession
from fastapi import APIRouter, HTTPException, Query, Body
from conversation.conversation_history_helper import ConversationHistoryHelper

router = APIRouter(prefix="/chat", tags=["chat-history"])

# Existing response model
class ChatSessionResponse(BaseModel):
    user_id: str
    sessions: List[ChatSession]

# New response models for session history
class ConversationMessage(BaseModel):
    request_id: str
    user_input: str 
    response: str

class SessionHistoryResponse(BaseModel):
    user_id: str
    session_id: str
    messages: List[ConversationMessage]

# New model for like/unlike request
class LikeRequest(BaseModel):
    liked: bool

# Existing endpoint
@router.get("/histories/{user_id}", response_model=ChatSessionResponse)
def get_chat_histories(user_id: str):
    try:
        helper = ConversationHistoryHelper()  # Assuming it's not a static class
        sessions = helper.get_session_list(user_id)
        return ChatSessionResponse(
            user_id=user_id,
            sessions=sessions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for session history
@router.get("/histories/{user_id}/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(
    user_id: str,
    session_id: str,
    limit: int = Query(default=10, gt=0)
):
    try:
        helper = ConversationHistoryHelper()
        histories = helper.get_conversation_history(user_id, session_id, limit)
        messages = [
            ConversationMessage(
                request_id=msg.request_id,
                user_input=msg.user_input,
                response=msg.response
            ) for msg in histories
        ]
        
        return SessionHistoryResponse(
            user_id=user_id,
            session_id=session_id,
            messages=messages
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            }
        )

# New endpoint for like/unlike
@router.put("/histories/{user_id}/{session_id}/{request_id}/like", response_model=ConversationMessage)
def update_message_like(
    user_id: str,
    session_id: str,
    request_id: str,
    like_request: LikeRequest = Body(...)
):
    try:
        helper = ConversationHistoryHelper()
        updated_message = helper.update_message_like(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            liked=like_request.liked
        )
        
        if not updated_message:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_message": "Message not found",
                    "error_code": "NOT_FOUND"
                }
            )
            
        return ConversationMessage(
            request_id=updated_message.request_id,
            user_input=updated_message.user_input,
            response=updated_message.response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            }
        )