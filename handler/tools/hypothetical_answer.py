import traceback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from utils.logging_util import logger


class HypotheticalAnswerGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logger

    def generate(self, query: str) -> str:
        """Generate a hypothetical answer to help with document retrieval"""
        prompt = f"""You are assisting with document search optimization. Generate a hypothetical answer that contains likely relevant terms and concepts to help find matching documents.

        Question: "{query}"

        Requirements for the search-optimized answer:
        1. CONTENT:
           - Must contain key terms and concepts related to the question
           - Must include likely relevant terminology and phrases
           - Must match the topic and scope of the question
           - Must reflect common ways this topic is discussed
           - Must incorporate probable related concepts

        2. STRUCTURE:
           - Exactly 2-3 complete sentences
           - First sentence addresses main search terms
           - Second sentence expands with related concepts
           - Use natural, descriptive language
           - Include specific terms when the question contains them

        3. SEARCH OPTIMIZATION:
           - Include synonyms of key concepts
           - Use standard terminology for the topic
           - Incorporate common related terms
           - Mirror the language level of the question
           - Include likely associated concepts

        Format Requirements:
        - Direct response style
        - No hypothetical markers (e.g., "might be", "could be")
        - No meta-commentary
        - No lists or bullet points
        - No source citations or references

        Purpose: This answer will be used to find relevant documents, not as a final response to the user.
        Return only the search-optimized answer:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            cleaned_response = self._clean_response(response)
            return cleaned_response
        except Exception as e:
            self.logger.error(f"Error generating search-optimized answer: {str(e)}, stack: {traceback.format_exc()}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean and validate the response"""
        try:
            # Remove any common prefixes that LLMs might add
            prefixes_to_remove = [
                "Here's the answer:",
                "Answer:",
                "Response:",
                "Hypothetical answer:",
                "Search-optimized answer:"
            ]
            
            cleaned = response.strip()
            for prefix in prefixes_to_remove:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()

            # Remove any quotes that might have been added
            cleaned = cleaned.strip('"\'')

            # Validate response length (roughly 2-3 sentences)
            sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
            if len(sentences) < 2 or len(sentences) > 4:
                self.logger.warning(f"Response has incorrect number of sentences: {len(sentences)}")
                return cleaned  # Return anyway, but log the warning

            return cleaned

        except Exception as e:
            self.logger.error(f"Error cleaning response: {str(e)}, stack: {traceback.format_exc()}")
            return response.strip()
