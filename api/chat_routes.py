from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from config.common_settings import CommonConfig
from handler.generic_query_handler import QueryHandler
from utils.id_util import get_id

router = APIRouter(tags=['chat'])
base_config = CommonConfig()


query_handler = QueryHandler(
    llm=base_config.get_model("chatllm"),
    vector_store=base_config.get_vector_store(),
    config=base_config
)

class QueryRequest(BaseModel):
    user_input: str

class QueryResponse(BaseModel):
    data: Any
    user_input: str

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    x_user_id: str = Header(...),
    x_session_id: str = Header(...),
    authorization: str | None = Header(default=None)
):
    try:
        result = query_handler.handle(
            user_input=request.user_input,
            user_id=x_user_id,
            session_id=x_session_id,
            request_id=get_id()
        )
        return QueryResponse(
            data=result,
            user_input=request.user_input
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "data": None,
                "user_input": request.user_input,
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            }
        )

if __name__ == "__main__":
    response = query_handler.handle(user_input="What is the capital of France?", user_id="test", session_id="test",
                                  request_id=get_id())
    print(response)