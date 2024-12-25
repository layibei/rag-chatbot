from fastapi import FastAPI, Request
from langchain.globals import set_debug

from api.chat_history_routes import router as chat_history_router
from api.chat_routes import router as chat_router
from api.embedding_routes import router as embedding_router
from config.common_settings import CommonConfig
from preprocess.doc_embedding_job import DocEmbeddingJob
from utils.logging_util import logger, set_context, clear_context

set_debug(True)
app = FastAPI()

@app.middleware("http")
async def add_context(request: Request, call_next):
    """Add context information to logs for each request"""
    # Get context from headers
    user_id = request.headers.get('X-User-Id', 'unknown')
    session_id = request.headers.get('X-Session-Id', 'unknown')
    request_id = request.headers.get('X-Request-Id', 'unknown')
    
    # Set context for logging
    set_context('user_id', user_id)
    set_context('session_id', session_id)
    set_context('request_id', request_id)
    
    try:
        response = await call_next(request)
    finally:
        clear_context()
    
    return response

app.include_router(chat_history_router, prefix="/chat")
app.include_router(embedding_router, prefix="/embedding")
app.include_router(chat_router, prefix="/chat")

base_config = CommonConfig()
base_config.setup_proxy()

# init scheduled job
doc_embeddings_processor = DocEmbeddingJob()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
