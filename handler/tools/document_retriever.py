from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Dict, Any

from langchain_postgres import PGVector

from config.common_settings import CommonConfig
from utils.logging_util import logger


class DocumentRetriever:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logger
        self.reranker = config.get_model("rerank")
        self.rerank_enabled = config.get_query_config("search.rerank_enabled", False)

    def run(self, query: str, relevance_threshold: float = 0.7, max_documents: int = 5) -> List[Document]:
        """
        Retrieve relevant documents with similarity scoring and filtering.
        
        Args:
            query: The search query
            relevance_threshold: Minimum similarity score (0.0 to 1.0)
            max_documents: Maximum number of documents to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Get more candidates for reranking
            k = max_documents * 2 if self.reranker and self.rerank_enabled  else max_documents * 1
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            candidates = []
            for doc, score in results:
                # Handle score based on vector store type
                if isinstance(self.vectorstore, PGVector):
                    # PGVector returns cosine similarity (-1 to 1)
                    similarity = (score + 1) / 2  # Convert to 0-1 range
                else:
                    # Assume cosine distance (0 to 2)
                    similarity = 1 - (score / 2)  # Convert to 0-1 range
                
                doc.metadata["vector_similarity"] = similarity
                candidates.append(doc)

            # Filter and sort by relevance
            relevant_docs = []
            for doc in candidates:
                # Convert similarity score to relevance threshold
                if doc.metadata["vector_similarity"] >= relevance_threshold:
                    # Add similarity score to metadata
                    doc.metadata["similarity_score"] = doc.metadata["vector_similarity"]
                    relevant_docs.append(doc)

            # Sort by similarity score and limit results
            relevant_docs.sort(
                key=lambda x: x.metadata["similarity_score"], 
                reverse=True
            )
            relevant_docs = relevant_docs[:max_documents]

            self.logger.info(
                f"Retrieved {len(relevant_docs)} relevant documents "
                f"(threshold: {relevance_threshold})"
            )

            return relevant_docs

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise
