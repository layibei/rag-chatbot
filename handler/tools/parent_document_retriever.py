import logging
import traceback
import os
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from config.common_settings import CommonConfig
from utils.logging_util import logger


class ParentChildDocumentRetriever:
    """
    A retriever that handles parent-child document relationships.
    It focuses on retrieving both parent and relevant child documents
    based on a query without the document addition functionality.
    """
    
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        """
        Initialize the ParentChildDocumentRetriever.
        
        Args:
            llm: The language model to use for reranking (if enabled)
            vectorstore: The vector store containing indexed documents
            config: Configuration object with retriever settings
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logger
        
        # Configure parent-child settings from config
        search_config = self.config.get_query_config("search")
        self.top_k_children = search_config.get("top_k_children", 2)
        self.parent_child_enabled = search_config.get("parent_child_enabled", False)
        
        # Only load these configurations if parent-child mode is enabled
        if self.parent_child_enabled:
            self.parent_chunk_size = search_config.get("parent_chunk_size", 2000)
            self.child_chunk_size = search_config.get("child_chunk_size", 400)
            self.chunk_overlap = search_config.get("chunk_overlap", 50)
            self.batch_size = search_config.get("batch_size", 32)
        
        # Configure reranking if enabled
        self.rerank_enabled = config.get_query_config("search.rerank_enabled", False)
        self.reranker = config.get_model("rerank") if self.rerank_enabled else None
        
    def _get_parent_ids_from_children(self, child_documents: List[Document]) -> List[Tuple[str, str]]:
        """
        Extracts parent document IDs and source information from child documents' metadata.
        
        Args:
            child_documents: List of child documents retrieved from a search
            
        Returns:
            List of tuples containing (parent_id, source)
        """
        parent_info = []
        
        for doc in child_documents:
            parent_id = doc.metadata.get("parent_id")
            source = doc.metadata.get("source", "")
            
            if parent_id and (parent_id, source) not in parent_info:
                parent_info.append((parent_id, source))
        
        return parent_info
    
    def _retrieve_parent_documents(self, parent_info: List[Tuple[str, str]]) -> List[Document]:
        """
        Retrieve parent documents using VectorStore's metadata filtering capabilities
        with both parent_id and source filters
        
        Args:
            parent_info: List of tuples containing (parent_id, source)
            
        Returns:
            List of retrieved parent documents
        """
        parent_documents = []
        
        try:
            self.logger.info(f"Retrieving parent documents with info: {parent_info}")
            
            # Detect vector store type
            vectorstore_type = type(self.vectorstore).__name__
            self.logger.info(f"Vector store type: {vectorstore_type}")
            
            # Qdrant-specific processing
            if "Qdrant" in vectorstore_type:
                self.logger.info("Using Qdrant-specific filtering format")
                
                try:
                    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                    
                    for parent_id, source in parent_info:
                        # Create Qdrant filter conditions
                        must_conditions = [
                            FieldCondition(
                                key="metadata.parent_id",
                                match=MatchValue(value=parent_id)
                            ),
                            FieldCondition(
                                key="metadata.is_parent",
                                match=MatchValue(value=True)
                            )
                        ]
                        
                        # Add source filter if available
                        if source:
                            must_conditions.append(
                                FieldCondition(
                                    key="metadata.source",
                                    match=MatchValue(value=source)
                                )
                            )
                        
                        filter_condition = Filter(must=must_conditions)
                        
                        # Use direct Qdrant client for search
                        search_results = self.vectorstore.client.search(
                            collection_name=self.vectorstore.collection_name,
                            query_vector=[0.0] * 1024,  # Use zero vector, rely only on filter conditions
                            query_filter=filter_condition,
                            limit=5,
                            with_payload=True
                        )
                        
                        if search_results:
                            self.logger.info(f"Found {len(search_results)} documents with parent_id={parent_id} and source={source} using Qdrant filter")
                            
                            for result in search_results:
                                # Convert to Document object
                                doc_content = result.payload.get("page_content", "")
                                doc_metadata = result.payload.get("metadata", {})
                                doc = Document(
                                    page_content=doc_content,
                                    metadata={**doc_metadata, "score": result.score, "id": result.id}
                                )
                                parent_documents.append(doc)
                        else:
                            self.logger.warning(f"No documents found with parent_id={parent_id} and source={source} using Qdrant filter")
                        
                    # If documents found, return immediately
                    if parent_documents:
                        return parent_documents
                        
                except Exception as e:
                    self.logger.error(f"Error using Qdrant-specific filtering method: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Generic LangChain method - try different filter formats
            self.logger.info("Trying generic LangChain filtering formats")
            
            for parent_id, source in parent_info:
                # Create base filter that will be expanded with source if available
                base_filter = {"parent_id": parent_id, "is_parent": True}
                if source:
                    base_filter["source"] = source
                
                # Try different filter formats
                filter_formats = [
                    {"filter": base_filter},
                    {"metadata": base_filter},
                    base_filter,
                    {"filter": {"metadata": base_filter}},
                ]
                
                # Add where clause format with metadata prefix for fields
                metadata_filter = {f"metadata.{k}": v for k, v in base_filter.items()}
                filter_formats.append({"where": metadata_filter})
                
                for i, filter_format in enumerate(filter_formats):
                    try:
                        self.logger.debug(f"Trying filter format {i+1}: {filter_format}")
                        docs = self.vectorstore.similarity_search(
                            query="",  # Empty query
                            k=5,
                            **filter_format
                        )
                        
                        if docs:
                            source_str = f"and source={source}" if source else ""
                            self.logger.info(f"Found {len(docs)} documents with parent_id={parent_id} {source_str} using format {i+1}")
                            parent_documents.extend(docs)
                            break  # Stop trying other formats after finding a working one
                    except Exception as e:
                        self.logger.debug(f"Format {i+1} failed: {str(e)}")
            
            # If still no documents found, try manual filtering
            if not parent_documents:
                self.logger.info("Trying direct search with manual filtering")
                try:
                    # Get more documents for manual filtering
                    all_docs = self.vectorstore.similarity_search(
                        query="",  # Empty query
                        k=100      # Get more documents to have enough samples for filtering
                    )
                    
                    # Filter for each parent_id and source
                    for parent_id, source in parent_info:
                        filtered_docs = [
                            doc for doc in all_docs 
                            if doc.metadata.get("parent_id") == parent_id and 
                               doc.metadata.get("is_parent") is True and
                               (not source or doc.metadata.get("source") == source)
                        ]
                        
                        if filtered_docs:
                            source_str = f"and source={source}" if source else ""
                            self.logger.info(f"Found {len(filtered_docs)} documents with parent_id={parent_id} {source_str} via manual filtering")
                            parent_documents.extend(filtered_docs)
                except Exception as e:
                    self.logger.error(f"Error during direct search and manual filtering: {str(e)}")
            
            # Finally try get_relevant_documents method as fallback
            if not parent_documents and hasattr(self.vectorstore, "get_relevant_documents"):
                self.logger.info("Trying get_relevant_documents method as final retrieval approach")
                for parent_id, source in parent_info:
                    try:
                        filters = {"doc_id": parent_id, "is_parent": True}
                        if source:
                            filters["source"] = source
                            
                        docs = self.vectorstore.get_relevant_documents(
                            "", metadata=filters
                        )
                        if docs:
                            source_str = f"and source={source}" if source else ""
                            self.logger.info(f"Found {len(docs)} documents with parent_id={parent_id} {source_str} using get_relevant_documents")
                            parent_documents.extend(docs)
                    except Exception as e:
                        self.logger.warning(f"Error using get_relevant_documents for retrieval: {str(e)}")
            
            # Final results
            if parent_documents:
                self.logger.info(f"Retrieved a total of {len(parent_documents)} parent documents")
            else:
                self.logger.warning(f"No parent documents found")
                
            return parent_documents
            
        except Exception as e:
            self.logger.error(f"Error retrieving parent documents: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return []
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks documents based on relevance to the query using the reranker model.
        
        Args:
            query: The user query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.rerank_enabled or not self.reranker or not documents:
            return documents
            
        try:
            pairs = []
            for doc in documents:
                pairs.append([query, doc.page_content])
                
            scores = self.reranker.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Combine documents with scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add scores to metadata
            for doc, score in scored_docs:
                doc.metadata["rerank_score"] = score
                
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return documents
    
    def run(self, query: str, relevance_threshold: float = 0.7, max_documents: int = 5, retrieve_only_children: bool = False) -> List[Document]:
        """
        Run retrieval for parent-child documents based on the query.
        
        Args:
            query: The user query
            relevance_threshold: Minimum relevance score threshold
            max_documents: Maximum number of documents to retrieve
            retrieve_only_children: When set to True, only return child documents without retrieving parent documents
            
        Returns:
            List of documents (either parent or child documents depending on retrieve_only_children), sorted by relevance
        """
        self.logger.info(f"Running parent-child document retrieval with query: {query}")
        
        if not self.parent_child_enabled:
            self.logger.warning("Parent-child retrieval is disabled in configuration")
            return []
            
        try:
            # First, retrieve the most relevant child documents
            # Use generic retrieval method and filter manually
            child_documents = self.vectorstore.similarity_search_with_score(
                query=query,
                k=max_documents 
            )
            
            # Extract and parse scores, only keeping child documents
            scored_children = []
            for doc, score in child_documents:
                # Only keep child documents, filter out parent documents
                if doc.metadata.get("is_parent") is not True:
                    doc.metadata["vector_score"] = score
                    scored_children.append(doc)
            
            
            
            # Print matched child documents or "not found" message
            if not scored_children:
                print("\n=== No matching child documents found ===\n")
            else:
                print(f"\n=== Found {len(scored_children)} matching child documents ===")
                for i, doc in enumerate(scored_children, 1):
                    print(f"\nChild Document {i}:")
                    print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                    score = doc.metadata.get("vector_score", "unknown")
                    print(f"Vector similarity score: {score}")
                    print(f"Parent document ID: {doc.metadata.get('parent_id', 'unknown')}")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                print("\n=== End of child documents list ===\n")
            
            # If only child documents are requested, return them directly
            if retrieve_only_children:
                self.logger.info("Returning only child documents as requested")
                
                # Filter by threshold if specified
                if relevance_threshold > 0:
                    scored_children = [
                        doc for doc in scored_children 
                        if doc.metadata.get("vector_score", 0) >= relevance_threshold
                    ]
                
                # Rerank child documents if enabled
                if self.rerank_enabled:
                    scored_children = self._rerank_documents(query, scored_children)
                
                # Limit to maximum number of documents
                return scored_children[:max_documents]
            
            if self.rerank_enabled and self.reranker and scored_children:
                self.logger.info(f"Reranking {len(scored_children)} child documents before parent retrieval")
                reranked_children = self._rerank_documents(query, scored_children)
                
                # just use top k
                top_children = reranked_children[:self.top_k_children]
                self.logger.info(f"Using top {len(top_children)} child documents after reranking")
                
                # print top k child documents
                print(f"\n=== Top {len(top_children)} child documents after reranking ===")
                for i, doc in enumerate(top_children, 1):
                    print(f"\nTop Child Document {i}:")
                    print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                    score = doc.metadata.get("rerank_score", "unknown")
                    print(f"Rerank score: {score}")
                    print(f"Parent document ID: {doc.metadata.get('parent_id', 'unknown')}")
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                print("\n=== End of top child documents ===\n")
                
                # use top k child documents to get parent document ids
                parent_info = self._get_parent_ids_from_children(top_children)
            else:
                # not rerank, use all child documents
                parent_info = self._get_parent_ids_from_children(scored_children)

            # Get parent document IDs and sources from child documents
            parent_info = self._get_parent_ids_from_children(scored_children)
            
            if not parent_info:
                self.logger.warning("No parent document IDs found from child documents")
                print("\n=== Could not find parent document IDs from child documents ===\n")
                return []
                
            # Retrieve parent documents
            parent_documents = self._retrieve_parent_documents(parent_info)
            
            if not parent_documents:
                self.logger.warning("Failed to retrieve parent documents")
                print("\n=== Failed to retrieve parent documents ===\n")
                return []
                
            # Rerank parent documents if reranking is enabled
            if self.rerank_enabled:
                parent_documents = self._rerank_documents(query, parent_documents)
                
            # Filter by threshold if specified
            if relevance_threshold > 0:
                # For reranked documents
                if self.rerank_enabled:
                    parent_documents = [
                        doc for doc in parent_documents 
                        if doc.metadata.get("rerank_score", 0) >= relevance_threshold
                    ]
                # For vector similarity scores
                else:
                    parent_documents = [
                        doc for doc in parent_documents 
                        if doc.metadata.get("vector_score", 0) >= relevance_threshold
                    ]
            
            # Limit to maximum number of documents
            return parent_documents[:max_documents]
            
        except Exception as e:
            self.logger.error(f"Error in parent-child document retrieval: {str(e)}")
            self.logger.debug(traceback.format_exc())
            print(f"\n=== Error during retrieval process: {str(e)} ===\n")
            raise

