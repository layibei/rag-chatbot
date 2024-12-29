from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langchain.globals import set_debug
from starlette.middleware.base import BaseHTTPMiddleware

from api.chat_history_routes import router as chat_history_router
from api.chat_routes import router as chat_router
from api.embedding_routes import router as embedding_router
from config.common_settings import CommonConfig
from preprocess.doc_embedding_job import DocEmbeddingJob
from utils.id_util import get_id
from utils.logging_util import logger, set_context, clear_context

# Global config instance
base_config = CommonConfig()


class LoggingContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get user_id from headers (required)
        user_id = request.headers.get('X-User-Id', 'unknown')
        
        # Get optional session_id and request_id from headers
        session_id = request.headers.get('X-Session-Id')
        request_id = request.headers.get('X-Request-Id')
        
        # For /query endpoints, generate missing IDs
        if request.url.path == '/chat/completion':
            headers_modified = False
            
            if not session_id:
                session_id = f"sess_{get_id().lower()}"
                # Modify request headers
                request.headers._list.append(
                    (b'x-session-id', session_id.encode())
                )
                headers_modified = True
                logger.debug(f"Generated new session_id: {session_id}")
            
            if not request_id:
                request_id = f"req_{get_id().lower()}"
                # Modify request headers
                request.headers._list.append(
                    (b'x-request-id', request_id.encode())
                )
                headers_modified = True
                logger.debug(f"Generated new request_id: {request_id}")
            
            if headers_modified:
                # Update the headers scope for ASGI
                request.scope['headers'] = request.headers._list
        else:
            # For non-query endpoints, use defaults if not provided
            session_id = session_id or 'unknown'
            request_id = request_id or 'unknown'

        # Set context using keyword arguments
        set_context(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )

        try:
            response = await call_next(request)
            return response
        finally:
            clear_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application"""
    embedding_job = None
    try:
        # Startup
        logger.info("Initializing application...")

        # Initialize config and setup proxy first
        proxy_result = await base_config.asetup_proxy()
        logger.info(f"Proxy setup {'enabled' if proxy_result else 'disabled'}")

        # Initialize other components (make this non-blocking)
        embedding_job = DocEmbeddingJob()
        init_result = await embedding_job.initialize()
        logger.info(f"Document embedding job initialization {'successful' if init_result else 'failed'}")

        logger.info("Application startup completed")

        # Important: yield here to let FastAPI take control
        yield

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        # Shutdown
        if embedding_job and embedding_job.scheduler:
            embedding_job.scheduler.shutdown()
        logger.info("Shutting down application...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(LoggingContextMiddleware)

app.include_router(chat_history_router, prefix="/chat")
app.include_router(embedding_router, prefix="/embedding")
app.include_router(chat_router, prefix="/chat")


if __name__ == "__main__":
    # set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
