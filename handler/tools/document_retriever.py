import traceback
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

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

        # Batch size for parallel processing
        self.batch_size = config.get_query_config("search.batch_size", 32)

        # Configure whether to use query expansion and hypothetical answers
        self.use_query_expansion = config.get_query_config("search.query_expansion_enabled", False)
        self.use_hypothetical = config.get_query_config("search.hypothetical_answer_enabled", False)

        # Initialize cross-encoder for better semantic reranking
        self.cross_encoder = self.config.get_model("rerank")
        # Get the actual tokenizer
        self.tokenizer = self.config.get_tokenizer()

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using model-based reranker or BM25"""
        try:
            if self.rerank_enabled and self.reranker:
                self.logger.debug("Using model-based reranker")
                return self._model_rerank(query, documents)
            else:
                return documents
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}, stack: {traceback.format_exc()}")
            return documents

    def _model_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using cross-encoder for better semantic matching"""
        try:
            # Get query tokens length
            query_tokens = self.tokenizer(query, add_special_tokens=False)['input_ids']
            query_length = len(query_tokens)

            # Calculate max tokens for document (512 - query_length - special_tokens)
            max_doc_tokens = 512 - query_length - 3  # [CLS], [SEP], [SEP]

            # Prepare pairs with token-aware truncation
            pairs = []
            for doc in documents:
                content = doc.page_content
                # Count actual tokens
                content_tokens = self.tokenizer(
                    content,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_doc_tokens
                )['input_ids']

                # Decode truncated tokens back to text
                truncated_content = self.tokenizer.decode(content_tokens)
                pairs.append([query, truncated_content])

            # Get semantic relevance scores
            scores = self.cross_encoder.predict(
                pairs,
                batch_size=32,
                show_progress_bar=True
            )

            # Combine documents with scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Log for debugging
            for doc, score in scored_docs:
                self.logger.debug(
                    f"Query: {query[:200]}...\n"
                    f"Content: {doc.page_content[:300]}...\n"
                    f"Score: {score:.3f}\n"
                )

            return [doc for doc, score in scored_docs if score > 0]

        except Exception as e:
            self.logger.error(f"Error during cross-encoder reranking: {str(e)}")
            return documents

    def _deduplicate_results(self, documents: List[Document]) -> List[Document]:
        """
        Deduplicate documents based on their IDs or content hash while preserving the highest scoring versions.
        """
        if not documents:
            return []

        unique_docs = {}

        for doc in documents:
            # Use document ID if available, otherwise fall back to content hash
            doc_id = doc.metadata.get("trunk_id", doc.metadata.get("content_hash"))

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

    def _batch_vector_search(self, queries: List[str], k: int) -> List[Document]:
        """Perform batched vector search for multiple queries"""
        all_results = []

        # Process queries in batches
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = []

            # Parallel vector search if supported by the vectorstore
            if hasattr(self.vectorstore, 'batch_similarity_search_with_score'):
                batch_results = self.vectorstore.batch_similarity_search_with_score(
                    queries=batch,
                    k=k
                )
            else:
                # Fallback to sequential processing
                for query in batch:
                    results = self.vectorstore.similarity_search_with_score(
                        query=query,
                        k=k
                    )
                    batch_results.extend(results)

            for doc, score in batch_results:
                doc.metadata["vector_score"] = score
                all_results.append(doc)

        return all_results

    def run(self, query: str, relevance_threshold: float = 0.7, max_documents: int = 5) -> List[Document]:

        try:
            self.logger.info(f"Running document retrieval with query: {query}")

            # Get configuration
            max_documents = self.config.get_query_config("search.top_k", max_documents)
            queries = [query]

            # 1. Optional Query Expansion
            if self.use_query_expansion:
                expanded_queries = self.query_expander.expand_query(query)
                queries.extend(expanded_queries)
                self.logger.debug(f"Expanded queries: {expanded_queries}")

            # 2. Efficient Batch Vector Search
            vector_results = self._batch_vector_search(queries, max_documents)

            # 3. Optional Graph Search (if enabled)
            if self.config.get_query_config("search.graph_search_enabled", False):
                graph_results = self.graph_store_helper.find_related_chunks(query, max_documents)
                vector_results.extend(graph_results)

            # 4. Optional Hypothetical Answer Search
            if self.use_hypothetical:
                hypothetical = self.hypothetical_generator.generate(query)
                if hypothetical:
                    hyp_results = self._batch_vector_search([hypothetical], max_documents)
                    vector_results.extend(hyp_results)

            # 5. Early deduplication to reduce reranking workload
            merged_results = self._deduplicate_results(vector_results)

            # 6. Rerank only if we have more documents than needed
            if len(merged_results) > max_documents:
                reranked_results = self._rerank_documents(query, merged_results)
            else:
                reranked_results = merged_results

            return reranked_results[:3]

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}, stack: {traceback.format_exc()}")
            raise


if __name__ == "__main__":
    # Example usage
    config = CommonConfig()
    vectorstore = config.get_vector_store()
    llm = config.get_model("chatllm")
    document_retriever = DocumentRetriever(llm, vectorstore, config)

    query = "Where is the capital of France?"
    documents = document_retriever._model_rerank(query, [Document(page_content="Paris is the capital of France."),
                                                         Document(page_content="Paris is a good place."),
                                                         Document(page_content="I love Beijing."),
                                                         Document(page_content="I travelled to Paris last Sumar."),
                                                         Document(page_content="Paris.")])
    print(documents)
