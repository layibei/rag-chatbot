from typing import List
from langchain_community.document_loaders import ConfluenceLoader as LangChainConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
import requests
from base64 import b64encode

from preprocess.loader.base_loader import DocumentLoader

class ConfluenceLoader(DocumentLoader):
    def load(self, url: str) -> List[Document]:
        """Override default load method to handle Confluence URLs"""
        try:
            self.logger.info(f"Loading Confluence page: {url}")
            loader = self.get_loader(url)
            if not loader:
                self.logger.error(f"Failed to create loader for Confluence URL: {url}")
                raise ValueError(f"Failed to create loader for Confluence URL: {url}")
            
            # Extract page ID from URL
            page_id = self._extract_page_id(url)
            self.logger.info(f"Extracted page id:{page_id} from {url}.")
            if not page_id:
                raise ValueError(f"Could not extract page ID from URL: {url}")
            
            # Load specific page
            documents = loader.load(page_ids=[page_id])
            if not documents:
                self.logger.error(f"Loaded content is empty for Confluence page: {url}")
                raise ValueError(f"Loaded content is empty for Confluence page: {url}")

            splitter = self.get_splitter(documents)
            if not splitter:
                self.logger.error(f"Failed to create splitter for Confluence page: {url}")
                raise ValueError(f"Failed to create splitter for Confluence page: {url}")

            return splitter.split_documents(documents)
        except Exception as e:
            self.logger.error(f"Failed to load Confluence page: {url}, Error: {str(e)}")
            raise

    def get_loader(self, url: str) -> BaseLoader:
        confluence_url = self.base_config.get_embedding_config("confluence.url")
        username = self.base_config.get_embedding_config("confluence.username")
        api_key = self.base_config.get_embedding_config("confluence.api_key")
        
        return LangChainConfluenceLoader(
            url=confluence_url,
            username=username,
            api_key=api_key
        )

    def get_splitter(self, documents: List[Document]) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.get_trunk_size(),
            chunk_overlap=self.get_overlap()
        )

    def is_supported_file_extension(self, file_path: str) -> bool:
        # Confluence pages don't have file extensions to check
        return True

    def _extract_page_id(self, url: str) -> str:
        """Extract page ID from Confluence URL"""
        try:
            parsed = urlparse(url)
            
            # 1. Check query parameter 'pageId'
            if 'pageId' in parsed.query:
                return parse_qs(parsed.query)['pageId'][0]
            
            # 2. Check for pages/viewpage.action?pageId format
            if 'viewpage.action' in parsed.path and 'pageId' in parsed.query:
                return parse_qs(parsed.query)['pageId'][0]
            
            # 3. Check for wiki/spaces format (Cloud)
            if '/wiki/spaces/' in parsed.path:
                path_segments = parsed.path.split('/')
                try:
                    # Find the 'pages' index and get the next segment
                    pages_index = path_segments.index('pages')
                    if len(path_segments) > pages_index + 1:
                        page_id = path_segments[pages_index + 1]
                        if page_id.isdigit():
                            return page_id
                except ValueError:
                    pass

            # 4. Check for display/SPACE/Page+Title format
            if '/display/' in parsed.path:
                # Use the Confluence API to look up page by title
                space_key = parsed.path.split('/display/')[1].split('/')[0]
                page_title = parsed.path.split('/')[-1].replace('+', ' ')
                
                confluence_url = self.base_config.get_embedding_config("confluence.url")
                username = self.base_config.get_embedding_config("confluence.username")
                api_key = self.base_config.get_embedding_config("confluence.api_key")
                
                # Use Confluence REST API to get page ID by title
                api_url = f"{confluence_url}/rest/api/content"
                params = {
                    "spaceKey": space_key,
                    "title": page_title,
                    "expand": "version"
                }
                headers = {
                    "Authorization": f"Basic {b64encode(f'{username}:{api_key}'.encode()).decode()}"
                }
                
                response = requests.get(api_url, params=params, headers=headers)
                if response.status_code == 200:
                    results = response.json()['results']
                    if results:
                        return results[0]['id']
                    
            # 5. Check numeric ID at end of path
            path_parts = parsed.path.split('/')
            if path_parts[-1].isdigit():
                return path_parts[-1]
            
            # 6. Check for pages/{pageId} format
            if '/pages/' in parsed.path:
                page_segment = parsed.path.split('/pages/')[-1]
                if '/' in page_segment:
                    potential_id = page_segment.split('/')[1]
                    if potential_id.isdigit():
                        return potential_id

            self.logger.error(f"Could not extract page ID from URL using any known format: {url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract page ID from URL: {url}, Error: {str(e)}")
            raise ValueError(f"Invalid Confluence URL format: {url}") 