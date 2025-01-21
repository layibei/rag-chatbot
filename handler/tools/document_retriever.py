import re
import traceback
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi

from config.common_settings import CommonConfig
from handler.store.graph_store_helper import GraphStoreHelper
from handler.tools.hypothetical_answer import HypotheticalAnswerGenerator
from handler.tools.query_expander import QueryExpander
from utils.logging_util import logger


class DocumentRetriever:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logger
        self.query_expander = QueryExpander(llm)
        self.hypothetical_generator = HypotheticalAnswerGenerator(llm)
        self.nlp = self.config.get_nlp_spacy()

        if self.config.get_query_config("search.graph_search_enabled", False):
            self.graph_store = self.config.get_graph_store()
            self.graph_store_helper = GraphStoreHelper(self.graph_store, config)

        # Configure reranking weights based on reranker type
        self.rerank_enabled = config.get_query_config("search.rerank_enabled", False)
        self.reranker = config.get_model("rerank") if self.rerank_enabled else None

        # Set weights based on reranker type
        if self.rerank_enabled and self.reranker:
            # Check if using a strong reranker (BGE or similar)
            reranker_name = str(self.reranker.__class__.__name__).lower()
            is_strong_reranker = any(name in reranker_name for name in ['bge', 'e5', 'bert'])

            # Default weights for strong rerankers (0.7 rerank, 0.3 vector)
            default_rerank_weight = 0.7 if is_strong_reranker else 0.5
            default_vector_weight = 1.0 - default_rerank_weight
        else:
            # For BM25 reranking
            default_rerank_weight = 0.5
            default_vector_weight = 0.5

        # Allow override through config
        self.rerank_weight = config.get_query_config("search.rerank_weight", default_rerank_weight)
        self.vector_weight = config.get_query_config("search.vector_weight", default_vector_weight)

        # Validate weights
        if abs(self.vector_weight + self.rerank_weight - 1.0) > 1e-6:
            self.logger.warning(
                f"Reranking weights should sum to 1.0. Current sum: {self.vector_weight + self.rerank_weight}. "
                f"Using reranker: {self.reranker.__class__.__name__ if self.reranker else 'BM25'}"
            )

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using model-based reranker or BM25"""
        try:
            if self.rerank_enabled and self.reranker:
                self.logger.debug("Using model-based reranker")
                return self._model_rerank(query, documents)
            else:
                self.logger.debug("Using BM25 reranker")
                return self._bm25_rerank(query, documents)
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}, stack: {traceback.format_exc()}")
            return documents

    def _model_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using the configured reranker model"""
        try:
            # Use plain text for reranking if available
            passages = [
                doc.metadata.get("plain_text", doc.page_content)
                for doc in documents
            ]

            # Get reranking scores
            rerank_scores = self.reranker.compute_score(query, passages)

            # Update document metadata with optimized weights
            for doc, rerank_score in zip(documents, rerank_scores):
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata.update({
                    "relevance_score": vector_score * self.vector_weight + rerank_score * self.rerank_weight,
                    "rerank_score": rerank_score,
                    "vector_score": vector_score
                })

            # Sort by combined score
            documents.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)
            return documents

        except Exception as e:
            self.logger.error(f"Error during model reranking: {str(e)}")
            return documents

    def _bm25_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using BM25 algorithm"""
        try:
            # Tokenize documents and query
            tokenized_docs = [self._tokenize_text(doc.page_content) for doc in documents]
            tokenized_query = self._tokenize_text(query)

            if not tokenized_docs:
                return documents

            # Create BM25 model
            bm25 = BM25Okapi(tokenized_docs)

            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)

            # Normalize scores to 0-1 range
            if len(bm25_scores) > 0:
                min_score = min(bm25_scores)
                max_score = max(bm25_scores)
                if max_score > min_score:
                    normalized_scores = [(score - min_score) / (max_score - min_score)
                                         for score in bm25_scores]
                else:
                    normalized_scores = [1.0 for _ in bm25_scores]
            else:
                normalized_scores = []

            # Update document metadata with optimized weights
            for doc, bm25_score in zip(documents, normalized_scores):
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata.update({
                    "relevance_score": vector_score * self.vector_weight + bm25_score * self.rerank_weight,
                    "rerank_score": bm25_score,
                    "vector_score": vector_score
                })

            # Sort by combined score
            documents.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)
            return documents

        except Exception as e:
            self.logger.error(f"Error during BM25 reranking: {str(e)}")
            return documents

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization"""
        try:
            if not text:
                return []

            # Process with spaCy
            doc = self.nlp(text)

            # Extract meaningful tokens while filtering out stop words and punctuation
            tokens = [
                token.lemma_.lower() for token in doc
                if not (token.is_stop or token.is_punct or token.is_space)
                   and len(token.text.strip()) > 1
            ]

            return tokens
        except Exception as e:
            self.logger.error(f"Error during tokenization: {str(e)}, stack:{traceback.format_exc()}")
            return self._fallback_tokenize_text(text)

    def _fallback_tokenize_text(self, text: str) -> List[str]:
        """
        Enhanced tokenization for BM25 with better text preprocessing
        """
        try:
            if not text:
                return []

            # Convert to lowercase
            text = text.lower()

            # Replace common contractions
            contractions = {
                "n't": " not",
                "'s": " is",
                "'m": " am",
                "'re": " are",
                "'ll": " will",
                "'ve": " have",
                "'d": " would"
            }
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)

            # Remove special characters but keep important punctuation
            text = re.sub(r'[^a-z0-9\s\-\.]', ' ', text)

            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)

            # Split on whitespace
            tokens = text.split()

            # Remove very short tokens and common stop words (optional)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
            tokens = [
                token for token in tokens
                if len(token) > 1 and token not in stop_words
            ]

            # Handle numbers and special cases
            processed_tokens = []
            for token in tokens:
                # Keep numbers intact but remove standalone punctuation
                if token.replace('.', '').replace('-', '').isdigit():
                    processed_tokens.append(token)
                # Split hyphenated words
                elif '-' in token:
                    processed_tokens.extend(token.split('-'))
                else:
                    processed_tokens.append(token)

            return processed_tokens

        except Exception as e:
            self.logger.warning(f"Error in tokenization: {str(e)}")
            # Fallback to basic tokenization
            return text.lower().split()

    def _deduplicate_results(self, documents: List[Document]) -> List[Document]:
        """
        Deduplicate documents based on their IDs or content hash while preserving the highest scoring versions.
        """
        if not documents:
            return []

        unique_docs = {}
        
        for doc in documents:
            # Use document ID if available, otherwise fall back to content hash
            doc_id = getattr(doc, 'id', hash(doc.page_content))
            
            if doc_id in unique_docs:
                # Keep the version with the higher score
                existing_score = unique_docs[doc_id].metadata.get("vector_score", 0.0)
                current_score = doc.metadata.get("vector_score", 0.0)
                
                if current_score > existing_score:
                    unique_docs[doc_id] = doc
            else:
                unique_docs[doc_id] = doc

        # Convert back to list and sort by score
        deduplicated_docs = list(unique_docs.values())
        deduplicated_docs.sort(
            key=lambda x: x.metadata.get("vector_score", 0.0),
            reverse=True
        )
        
        self.logger.debug(
            f"Deduplication: {len(documents)} inputs -> {len(deduplicated_docs)} unique documents"
        )
        
        return deduplicated_docs

    def run(self, query: str, relevance_threshold: float = 0.7, max_documents: int = 5) -> List[Document]:
        try:
            self.logger.info(f"Running document retrieval with query: {query}")
            # 1. Query Expansion
            expanded_queries = self.query_expander.expand_query(query)
            self.logger.info(f"Expanded queries: {expanded_queries}")
            all_queries = [query] + expanded_queries
            # get top_k from config, default to max_documents
            max_documents = self.config.get_query_config("search.top_k", max_documents)

            # 2. Vector Search with all queries
            vector_results = []
            for q in all_queries:
                results = self.vectorstore.similarity_search_with_score(
                    query=q,
                    k=max_documents
                )
                # Store vector scores in metadata
                for doc, score in results:
                    doc.metadata["vector_score"] = score
                    vector_results.append(doc)

            # 3. Graph Database Search (if enabled)
            if self.config.get_query_config("search.graph_search_enabled", False):
                graph_results = self.graph_store_helper.find_related_chunks(query, max_documents)
                self.logger.debug(f"Graph search results: {graph_results}")
                vector_results.extend(graph_results)

            # 4. Hypothetical Answer Search (if enabled)
            if self.config.get_query_config("search.hypothetical_answer_enabled", False):
                hypothetical = self.hypothetical_generator.generate(query)
                self.logger.debug(f"Hypothetical answer: {hypothetical}")
                if hypothetical:
                    hyp_results = self.vectorstore.similarity_search_with_score(
                        query=hypothetical,
                        k=max_documents
                    )
                    # Store vector scores in metadata
                    for doc, score in hyp_results:
                        doc.metadata["vector_score"] = score
                        vector_results.append(doc)

            # 5. Merge and deduplicate results
            merged_results = self._deduplicate_results(vector_results)

            # 6. Rerank using appropriate method
            reranked_results = self._rerank_documents(query, merged_results)

            # 7. Filter and return top results
            final_results = [
                                doc for doc in reranked_results
                                if doc.metadata.get("relevance_score", 0) >= relevance_threshold
                            ][:max_documents]

            self.logger.info(
                f"Retrieved {len(final_results)} relevant documents "
                f"(threshold: {relevance_threshold}, reranking: {self.rerank_enabled})"
            )

            return final_results

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}, stack: {traceback.format_exc()}")
            raise
