import os

import dotenv
from fastapi import FastAPI
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.embeddings import HuggingFaceEmbeddings, SparkLLMTextEmbeddings
from langchain_community.llms.sparkllm import SparkLLM
from langchain_qdrant import QdrantVectorStore

from embedding.doc_embeddings import DocEmbeddings

dotenv.load_dotenv()
app = FastAPI()


class RagService:
    def __init__(self):
        self.llm = llm = SparkLLM()
        self.llm_chat = ChatSparkLLM()
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        # self.embeddings = SparkLLMTextEmbeddings()
        self.qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name="rag_docs",
            url="http://localhost:6333",
        )

        self.__docEmbeddings = DocEmbeddings(self.embeddings, self.qdrant)
        self.__docEmbeddings.load_documents("./data")

    def query(self, query):
        return self.llm.invoke(query, top_k=3, vector_store=self.qdrant)


if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    rag_service = RagService()
    # rag_service.query("Where is Xian?")
