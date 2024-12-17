from typing import TypedDict, Annotated, Any, Optional
from datetime import datetime
from langchain_core.documents import Document


class RequestState(TypedDict):
    user_id: str
    session_id: str
    request_id: str
    user_input: str
    response: Any
    source: str
    documents: list[Document]
    original_query: str
    transform_query_count: int
    web_search_count: int
    is_transformed_query: bool
    final_response: Any
    messages: list[dict]
    token_usage: dict

