from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
import re

from langchain_postgres import PGVector
from langchain_qdrant import QdrantVectorStore

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
        Retrieve relevant documents with similarity scoring, reranking and filtering.
        
        Args:
            query: The search query
            relevance_threshold: Minimum similarity score (0.0 to 1.0)
            max_documents: Maximum number of documents to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Get more candidates if reranking is enabled
            k = max_documents * 3 if self.rerank_enabled else max_documents * 2
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            # Process vector similarity scores
            candidates = self._process_similarity_scores(results)
            
            # Apply reranking if enabled
            if self.rerank_enabled and self.reranker:
                candidates = self._rerank_documents(query, candidates)
            elif len(candidates) > max_documents:
                # Apply BM25 scoring if no reranker but have extra candidates
                candidates = self._bm25_rerank(query, candidates)

            # Filter and sort
            relevant_docs = [
                doc for doc in candidates 
                if doc.metadata.get("relevance_score", 0) >= relevance_threshold
            ]
            relevant_docs.sort(
                key=lambda x: x.metadata.get("relevance_score", 0), 
                reverse=True
            )

            # Limit results
            relevant_docs = relevant_docs[:max_documents]
            
            self.logger.info(
                f"Retrieved {len(relevant_docs)} relevant documents "
                f"(threshold: {relevance_threshold}, reranking: {self.rerank_enabled})"
            )

            return relevant_docs

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def _process_similarity_scores(self, results: List[Tuple[Document, float]]) -> List[Document]:
        """Process and normalize vector similarity scores based on vector store type"""
        processed_docs = []
        
        for doc, score in results:
            # Calculate similarity score
            if isinstance(self.vectorstore, PGVector):
                similarity = (score + 1) / 2
            elif isinstance(self.vectorstore, QdrantVectorStore):
                similarity = score
            else:
                similarity = 1 - (score / 2)
            
            # Use markdown content for LLM if available
            if "markdown_content" in doc.metadata:
                # Store plain text version for vector operations
                #doc.metadata["plain_text"] = doc.page_content
                # Use markdown for LLM interactions
                doc.page_content = doc.metadata["markdown_content"]
            
            doc.metadata["relevance_score"] = similarity
            processed_docs.append(doc)
            
        return processed_docs


    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using the reranker model"""
        try:
            # Use plain text for reranking if available
            passages = [
                doc.metadata.get("plain_text", doc.page_content) 
                for doc in documents
            ]
            
            # Get reranking scores
            rerank_scores = self.reranker.compute_score(
                query,
                passages
            )

            # Update document metadata
            for doc, rerank_score in zip(documents, rerank_scores):
                vector_score = doc.metadata["relevance_score"]
                doc.metadata.update({
                    "relevance_score": vector_score * 0.3 + rerank_score * 0.7,
                    "rerank_score": rerank_score,
                    "vector_score": vector_score
                })

            return documents

        except Exception as e:
            self.logger.error(f"Error during document reranking: {str(e)}")
            return documents

    def _bm25_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using BM25 scoring algorithm"""
        try:
            # Tokenize documents and query
            doc_texts = [doc.page_content.lower().split() for doc in documents]
            query_tokens = query.lower().split()
            
            # Create BM25 model
            bm25 = BM25Okapi(doc_texts)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores to 0-1 range
            max_score = max(bm25_scores) if bm25_scores else 1
            normalized_scores = [score/max_score for score in bm25_scores]
            
            # Update document metadata
            for doc, bm25_score in zip(documents, normalized_scores):
                vector_score = doc.metadata["relevance_score"]
                # Combine scores with equal weights
                doc.metadata.update({
                    "relevance_score": (vector_score + bm25_score) / 2,
                    "bm25_score": bm25_score,
                    "vector_score": vector_score
                })
            
            self.logger.debug(f"BM25 reranking completed for {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error during BM25 reranking: {str(e)}")
            return documents
