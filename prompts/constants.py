from enum import Enum
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Enum for all available prompt templates"""
    GENERATE_RESPONSE = "generate_response.txt"
    REWRITE_QUERY = "rewrite_query.txt"
    GRADE_RESPONSE = "grade_response.txt"
    GENERATE_QUESTIONS = "generate_questions.txt"
    FORMAT_RESPONSE = "format_response.txt"

class PromptManager:
    """Manages loading and caching of prompt templates"""
    
    _instance = None
    _prompts: Dict[PromptTemplate, str] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of PromptManager"""
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the prompt manager and load all prompts"""
        self._prompts = {}
        self._prompt_dir = Path(__file__).parent
        
    def get_prompt(self, template: PromptTemplate) -> str:
        """
        Get a prompt template by its identifier.
        Loads from cache if available, otherwise loads from file.
        
        Args:
            template: The PromptTemplate enum value
            
        Returns:
            The prompt template string
            
        Raises:
            FileNotFoundError: If prompt template file doesn't exist
            ValueError: If template is invalid
        """
        if template not in self._prompts:
            try:
                prompt_path = self._prompt_dir / template.value
                if not prompt_path.exists():
                    raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
                
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self._prompts[template] = f.read()
                logger.debug(f"Loaded prompt template: {template.name}")
                
            except Exception as e:
                logger.error(f"Error loading prompt template {template.name}: {str(e)}")
                raise
                
        return self._prompts[template]
    
    def format_prompt(self, template: PromptTemplate, **kwargs) -> str:
        """
        Get and format a prompt template with the provided arguments.
        
        Args:
            template: The PromptTemplate enum value
            **kwargs: Format arguments for the template
            
        Returns:
            The formatted prompt string
        """
        prompt_template = self.get_prompt(template)
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required argument for prompt {template.name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error formatting prompt {template.name}: {str(e)}")
            raise

    def reload_prompt(self, template: PromptTemplate) -> None:
        """
        Force reload a prompt template from file.
        
        Args:
            template: The PromptTemplate enum value to reload
        """
        if template in self._prompts:
            del self._prompts[template]
        self.get_prompt(template)
        
    def reload_all(self) -> None:
        """Force reload all prompt templates"""
        self._prompts.clear() 