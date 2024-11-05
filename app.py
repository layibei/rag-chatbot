import asyncio
import os

import dotenv
from fastapi import FastAPI
from langchain.globals import set_debug
from langchain_community.chat_models import ChatSparkLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.sparkllm import SparkLLM
from langchain_postgres import PGVector
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisConfig, RedisVectorStore

from config.common_settings import CommonConfig
from handler.generic_query_handler import QueryHandler
from preprocess.index_log_helper import IndexLogHelper
from utils.logging_util import logger

from preprocess.doc_embeddings import DocEmbeddingsProcessor

dotenv.load_dotenv()
set_debug(True)
app = FastAPI()

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
postgres_uri = os.environ["POSTGRES_URI"]
# vector_store = PGVector(
#     embeddings=embeddings,
#     collection_name=collection_name,
#     connection=postgres_uri,
#     use_jsonb=True,
# )

config = RedisConfig(
    index_name="rag_docs",
    redis_url=os.environ["REDIS_URL"],
    distance_metric="COSINE",  # Options: COSINE, L2, IP
)
vector_store = RedisVectorStore(embeddings, config=config)

indexLogHelper = IndexLogHelper(postgres_uri)
docEmbeddingsProcessor = DocEmbeddingsProcessor(embeddings, vector_store, indexLogHelper)
queryHandler = QueryHandler(llm_chat, vector_store)


async def preprocess():
    logger.info("Loading documents...")
    await docEmbeddingsProcessor.load_documents(base_config.get_embedding_config().get("input_path"))


@app.get("/query")
def query(query):
    return queryHandler.handle(query)


if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    asyncio.run(preprocess())

    from langchain.globals import set_debug
    set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
