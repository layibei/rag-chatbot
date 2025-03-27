from typing import List, Dict, Any, Optional
from enum import Enum
from langchain_community.tools import TavilySearchResults, GoogleSearchResults
from langchain_community.utilities import SerpAPIWrapper, DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper, \
    BingSearchAPIWrapper
from langchain_core.documents import Document

from config.common_settings import CommonConfig
from utils.logging_util import logger


class SearchProvider(Enum):
    TAVILY = "tavily"
    GOOGLE = "google"
    SERPAPI = "serpapi"
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"

class WebSearch:
    def __init__(self, config: CommonConfig):
        self.logger = logger
        self.config = config
        self.web_search_tool = None
        self.tokenizer = self.config.get_tokenizer()
        # Initialize cross-encoder for better semantic reranking
        self.cross_encoder = self.config.get_model("rerank")
        # Configure reranking
        self.rerank_enabled = config.get_query_config("search.rerank_enabled", False)

        if config.get_query_config("search.web_search_enabled", False):
            self.logger.info("Web search is enabled")
            provider = config.get_query_config("search.provider", "tavily").lower()
            self._initialize_search_tool(provider)
        else:
            self.logger.info("Web search is disabled")

    def _initialize_search_tool(self, provider: str):
        """Initialize the appropriate search tool based on configuration"""
        try:
            max_results = self.config.get_query_config("search.top_k", 5) * 2
            
            if provider == SearchProvider.GOOGLE.value:
                # First create the API wrapper
                search_wrapper = GoogleSearchAPIWrapper(
                    google_api_key=self.config.get_env_variable("GOOGLE_API_KEY"),
                    google_cse_id=self.config.get_env_variable("GOOGLE_CSE_ID"),
                )
                self.web_search_tool = GoogleSearchResults(
                    api_wrapper=search_wrapper,
                    k=max_results
                )
                
            elif provider == SearchProvider.SERPAPI.value:
                # SerpAPI (paid, supports multiple search engines)
                api_key = self.config.get_env_variable("SERPAPI_API_KEY")
                self.web_search_tool = SerpAPIWrapper(
                    serpapi_api_key=api_key,
                    k=max_results
                )
                
            elif provider == SearchProvider.DUCKDUCKGO.value:
                # DuckDuckGo (free, no API key required)
                self.web_search_tool = DuckDuckGoSearchAPIWrapper(
                    max_results=max_results
                )
                
            else:  # Default to Tavily
                api_key = self.config.get_env_variable("TAVILY_API_KEY")
                self.web_search_tool = TavilySearchResults(
                    tavily_api_key=api_key,
                    k=max_results,
                    max_results=max_results
                )
            
            self.logger.info(f"Initialized {provider} search provider")
            
        except Exception as e:
            self.logger.error(f"Error initializing search tool: {str(e)}")
            raise

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

    def run(self, query: str) -> List[Document]:
        """Execute web search with error handling and logging"""
        try:
            self.logger.info(f"Running web search for query: {query}")

            if not self.config.get_query_config("search.web_search_enabled", False):
                self.logger.info("Web search is disabled")
                return []

            if not self.web_search_tool:
                self.logger.error("Web search tool not initialized")
                return []

            max_results = self.config.get_query_config("limits.max_web_results", 3) * 2

            # Handle different search tool interfaces
            if isinstance(self.web_search_tool, (TavilySearchResults, GoogleSearchResults)):
                results = self.web_search_tool.invoke(query)
            elif isinstance(self.web_search_tool, DuckDuckGoSearchAPIWrapper):
                results = self.web_search_tool.results(query, max_results)
            elif isinstance(self.web_search_tool, (BingSearchAPIWrapper, SerpAPIWrapper)):
                results = self.web_search_tool.run(query)
            else:
                self.logger.error(f"Unsupported search tool type: {type(self.web_search_tool)}")
                return []
            
            # Convert results to LangChain Documents
            documents = self._normalize_results(results)

            if not documents:
                self.logger.warning(f"No results found for query: {query}")
                return []

            # Apply reranking if enabled
            if self.rerank_enabled and len(documents) > max_results:
                documents = self._model_rerank(query, documents)

            self.logger.info(f"Found {len(documents)} results for query")
            return documents[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error during web search: {str(e)}")
            return []

    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Document]:
        """Normalize results format to LangChain Documents"""
        normalized = []
        
        # Handle case where results might be a single dict
        if isinstance(results, dict):
            results = [results]
        
        for result in results:
            try:
                # Handle different result formats
                if isinstance(result, str):
                    # Some APIs might return plain text
                    normalized.append(Document(
                        page_content=result,
                        metadata={
                            'title': '',
                            'url': '',
                            'source': '',
                            'published_date': ''
                        }
                    ))
                else:
                    normalized.append(Document(
                        page_content=result.get('content', result.get('snippet', result.get('text', ''))),
                        metadata={
                            'title': result.get('title', ''),
                            'url': result.get('url', result.get('link', '')),
                            'source': result.get('source', ''),
                            'published_date': result.get('published_date', '')
                        }
                    ))
            except Exception as e:
                self.logger.error(f"Error normalizing result: {str(e)}")
                continue
        return normalized
