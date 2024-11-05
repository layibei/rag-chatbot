from typing import TypedDict, Annotated, Any

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class RequestState(TypedDict):
    user_id: str
    session_id: str
    request_id: str
    user_input: Annotated[list[str], add_messages]
    response: Any
    source: str
    documents: list[Document]
    original_query: str
    transform_query_count: int
    web_search_count: int
    is_transformed_query: bool
    final_response: Any

