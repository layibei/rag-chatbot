from typing import TypedDict, Annotated, Any, Dict, Union

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class RequestState(TypedDict):
    user_id: str
    session_id: str
    request_id: str
    user_input: str
    rewritten_query: str

    messages: Annotated[list[str], add_messages]
    response: Any
    documents: list[Document]
    web_results: Union[list[Document], list[Dict]]
    hallucination_risk: str
    confidence_score: str
    output_format: str

    suggested_questions: list[str] = []
    citations: list[str] = []

    # Attempt counters
    rewrite_attempts: int = 0
    web_search_attempts: int = 0
    enhance_attempts: int = 0
