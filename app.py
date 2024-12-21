import os

import dotenv
from fastapi import FastAPI
from langchain.globals import set_debug

from api.chat_routes import router as chat_router
from api.embedding_routes import router as embedding_router
from config.common_settings import CommonConfig
from config.database.database_manager import DatabaseManager
from conversation.repositories import ConversationHistoryRepository
from handler.generic_query_handler import QueryHandler
from preprocess.doc_embedding_job import DocEmbeddingJob
from preprocess.index_log.repositories import IndexLogRepository
from utils.lock.repositories import DistributedLockRepository

# from preprocess.doc_embeddings import DocEmbeddingsProcessor

dotenv.load_dotenv()
set_debug(True)
app = FastAPI()

app.include_router(embedding_router, prefix="/embedding")
app.include_router(chat_router, prefix="/chat")

base_config = CommonConfig()

llm = base_config.get_model("llm")
llm_chat = base_config.get_model("chatllm")
embeddings = base_config.get_model("embedding")
# embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
collection_name = "rag_docs"
# vector_store = QdrantVectorStore.from_existing_collection(
#     embedding=embeddings,
#     collection_name=collection_name,
#     api_key=os.environ["QDRANT_API_KEY"],
#     url="http://localhost:6333",
# )

# init scheduled job
doc_embeddings_processor = DocEmbeddingJob()

if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    from langchain.globals import set_debug

    set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
