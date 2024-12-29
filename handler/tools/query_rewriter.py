from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from utils.logging_util import logger


class QueryRewriter:
    """
    Improves query clarity and focus through intelligent rewriting.
    Makes queries more specific and searchable while preserving original intent.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logger

    def run(self, state: Dict[str, Any]) -> str:
        """Rewrite the query for better accuracy and searchability"""
        try:
            original_query = state.get("user_input", "")
            self.logger.info(f"Running query rewriter with user input: {original_query}")

            prompt = f"""You are an expert in query understanding and reformulation.

            Original Query: "{original_query}"

            Your task is to rewrite this query to be more specific while keeping the EXACT same intent.

            Guidelines:
            1. NEVER change the subject or topic
            2. Only add context that's clearly implied
            3. Keep the same type of question (what/how/why/when)
            4. Don't add unrelated concepts

            Examples:
            Query: "What's the latest Java version?"
            Better: "What is the current stable version of Java (JDK) as of 2024?"

            Query: ""What's the weather today in Xian China"
            Better: "What are the weather conditions like for today in Xi'an, China?"

            Rewrite the query following these guidelines. Return only the rewritten query, no explanations:"""

            rewritten_query = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            self.logger.info(f"Rewritten query: {rewritten_query}")

            # Validate the rewrite maintains topic
            if not self._validate_rewrite(original_query, rewritten_query):
                self.logger.warning("Rewrite validation failed, using original query")
                return original_query

            return rewritten_query

        except Exception as e:
            self.logger.error(f"Error rewriting query: {str(e)}")
            return original_query

    def _validate_rewrite(self, original: str, rewritten: str) -> bool:
        """Validate that the rewritten query maintains the original topic"""
        try:
            # Simple keyword overlap check
            original_words = set(original.lower().split())
            rewritten_words = set(rewritten.lower().split())
            
            # Remove common words
            stop_words = {'what', 'how', 'when', 'where', 'why', 'which', 'are', 'is', 
                         'the', 'to', 'in', 'on', 'at', 'and', 'or', 'for', 'of'}
            original_keywords = {w for w in original_words if w not in stop_words}
            
            # At least one key term should be preserved
            if not any(word in rewritten.lower() for word in original_keywords):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating rewrite: {str(e)}")
            return False
