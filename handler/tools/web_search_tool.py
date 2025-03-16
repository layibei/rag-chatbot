from typing import List, Dict, Any, Optional
from enum import Enum
from langchain_community.tools import TavilySearchResults, GoogleSearchResults
from langchain_community.utilities import SerpAPIWrapper, DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper

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

        if config.get_query_config("search.web_search_enabled", False):
            self.logger.info("Web search is enabled")
            provider = config.get_query_config("search.provider", "tavily").lower()
            self._initialize_search_tool(provider)
        else:
            self.logger.info("Web search is disabled")

    def _initialize_search_tool(self, provider: str):
        """Initialize the appropriate search tool based on configuration"""
        try:
            max_results = self.config.get_query_config("limits.max_web_results", 3) * 3
            
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

    def run(self, query: str) -> List[Dict[str, Any]]:
        """Execute web search with error handling and logging"""
        try:
            self.logger.info(f"Running web search for query: {query}")

            if not self.config.get_query_config("search.web_search_enabled", False):
                self.logger.info("Web search is disabled")
                return []

            if not self.web_search_tool:
                self.logger.error("Web search tool not initialized")
                return []

            max_results = self.config.get_query_config("limits.max_web_results", 3)

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
            
            # Normalize results format across different providers
            normalized_results = self._normalize_results(results)

            if not normalized_results:
                self.logger.warning(f"No results found for query: {query}")
                return []

            self.logger.info(f"Found {len(normalized_results)} results for query")
            return normalized_results
            
        except Exception as e:
            self.logger.error(f"Error during web search: {str(e)}")
            return []

    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize results format across different providers"""
        normalized = []
        
        # Handle case where results might be a single dict
        if isinstance(results, dict):
            results = [results]
        
        for result in results:
            try:
                # Handle different result formats
                if isinstance(result, str):
                    # Some APIs might return plain text
                    normalized.append({
                        'title': '',
                        'url': '',
                        'content': result,
                        'source': '',
                        'published_date': ''
                    })
                else:
                    normalized.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', result.get('link', '')),
                        'content': result.get('content', result.get('snippet', result.get('text', ''))),
                        'source': result.get('source', ''),
                        'published_date': result.get('published_date', '')
                    })
            except Exception as e:
                self.logger.error(f"Error normalizing result: {str(e)}")
                continue
        return normalized
