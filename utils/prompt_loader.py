from pathlib import Path
from typing import Optional
from langchain_core.prompts import PromptTemplate

def load_txt_prompt(file_path: str, input_variables: Optional[list] = None) -> PromptTemplate:
    """
    Load a prompt template from a txt file.
    
    Args:
        file_path: Path to the txt file
        input_variables: List of input variables in the template. If None, will try to detect from content.
    
    Returns:
        PromptTemplate: A LangChain prompt template
    """
    try:
        # Read the template from file
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        # If input_variables not provided, try to detect from template
        if input_variables is None:
            # Find all {variable} patterns in template
            import re
            input_variables = list(set(re.findall(r'\{(\w+)\}', template)))
            
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        
    except Exception as e:
        raise ValueError(f"Error loading prompt from {file_path}: {str(e)}") 