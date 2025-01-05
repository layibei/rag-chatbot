from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from utils.logging_util import logger
from config.common_settings import CommonConfig

class ResponseGrader:
    """Grades response quality and relevance to user query"""

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.llm = llm
        self.logger = logger
        self.config = config

    def run(self, state: Dict[str, Any]) -> float:
        """Grade response relevance and completeness"""
        self.logger.info("Grading response quality...")
        try:
            response = state.get("response", "")
            user_input = state.get("rewritten_query", "")
            
            grade = self._grade_response(response, user_input)
            
            self.logger.info(f"Response grade: {grade}")
            
            return grade

        except Exception as e:
            self.logger.error(f"Error grading response: {str(e)}")
            return {"needs_rewrite": True}

    def _grade_response(self, response: str, user_input: str) -> float:
        """Grade how well the response answers the user's question"""
        prompt = f"""As an expert response evaluator, grade how well this response answers the user's question.

        User Question: "{user_input}"
        Response: "{response}"

        GRADING CRITERIA:

        1. ANSWER RELEVANCE (50 points)
           - Direct answer to the question (30 points)
           - Appropriate level of detail (20 points)

        2. ANSWER COMPLETENESS (30 points)
           - Covers all aspects of the question (15 points)
           - No missing crucial information (15 points)

        3. ANSWER QUALITY (20 points)
           - Clarity and understandability (10 points)
           - Accuracy of information (10 points)

        SCORING INSTRUCTIONS:
        1. Score each category
        2. Sum all points
        3. Divide by 100 for final score

        Return ONLY the final decimal score between 0.0 and 1.0
        Example outputs: 0.95, 0.82, 0.67, 0.43"""

        try:
            response_text = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            import re
            if match := re.search(r'(\d+\.\d+)', response_text):
                score = float(match.group(1))
            else:
                score = float(response_text)
            
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            self.logger.error(f"Failed to parse response grade: {str(e)}")
            return 0.0 