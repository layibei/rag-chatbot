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

            Task: Analyze and rewrite this query to improve its clarity, specificity, and search effectiveness.

            Analysis Steps:
            1. Intent Understanding
               - Identify the core information need
               - Recognize implicit assumptions
               - Understand desired outcome type (explanation, comparison, procedure, etc.)

            2. Context Enrichment
               - Identify missing but implied context
               - Recognize relevant qualifiers
               - Consider temporal aspects if applicable

            3. Query Enhancement
               - Add essential qualifiers
               - Expand abbreviated terms
               - Include relevant synonyms
               - Make implicit concepts explicit

            4. Search Optimization
               - Use precise terminology
               - Include key concept variations
               - Maintain natural language structure
               - Ensure searchable patterns

            Rewriting Principles:
            ✓ Maintain original intent
            ✓ Increase specificity without overspecifying
            ✓ Preserve important constraints
            ✓ Remove ambiguity
            ✓ Keep natural language flow

            Examples:
            Query: "best time to plant"
            Better: "what is the optimal season and conditions for planting garden vegetables"

            Query: "difference between cold and flu"
            Better: "what are the key differences between common cold symptoms and influenza symptoms"

            Query: "how to make bread"
            Better: "what are the basic steps and ingredients needed for making homemade bread from scratch"

            Rewrite the query following these guidelines. Return only the rewritten query, no explanations or additional text:"""

            rewritten_query = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            self.logger.info(f"Re-written query is generated:{rewritten_query}")

            return rewritten_query

        except Exception as e:
            self.logger.error(f"Error rewriting query: {str(e)}")
            # Fallback to original query on error
            state["rewritten_query"] = state.get("user_input", "")
            return state
