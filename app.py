import os

import dotenv
from fastapi import FastAPI
from langchain.globals import set_debug

from api.chat_history_routes import router as chat_history_router
from api.chat_routes import router as chat_router
from api.embedding_routes import router as embedding_router
from config.common_settings import CommonConfig
from preprocess.doc_embedding_job import DocEmbeddingJob


dotenv.load_dotenv()
set_debug(True)
app = FastAPI()
app.include_router(chat_history_router, prefix="/chat")
app.include_router(embedding_router, prefix="/embedding")
app.include_router(chat_router, prefix="/chat")


# init scheduled job
doc_embeddings_processor = DocEmbeddingJob()

if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    from langchain.globals import set_debug

    set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
