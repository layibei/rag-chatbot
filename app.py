import asyncio
import os

import dotenv
from fastapi import FastAPI
from langchain.globals import set_debug
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.embeddings import HuggingFaceEmbeddings, SparkLLMTextEmbeddings
from langchain_community.llms.sparkllm import SparkLLM
from langchain_qdrant import QdrantVectorStore

from preprocess.index_log_helper import IndexLogHelper
from utils.logging_util import logger

from preprocess.doc_embeddings import DocEmbeddingsProcessor

dotenv.load_dotenv()
app = FastAPI()

llm = SparkLLM()
llm_chat = ChatSparkLLM()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="rag_docs",
    api_key=os.environ["QDRANT_API_KEY"],
    url="http://localhost:6333",
)
indexLogHelper = IndexLogHelper(os.environ["POSTGRES_URI"])
docEmbeddingsProcessor = DocEmbeddingsProcessor(embeddings, vector_store, indexLogHelper)


async def preprocess():
    logger.info("Loading documents...")
    await docEmbeddingsProcessor.load_documents("./data")


def query(self, query):
    return llm.invoke(query, top_k=3, vector_store=self.qdrant)


if __name__ == "__main__":
    set_debug(True)
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    asyncio.run(preprocess())
