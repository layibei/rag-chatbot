from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config.common_settings import CommonConfig
from utils.logging_util import logger


class HallucinationDetector:
    """Detects potential hallucinations in LLM responses"""

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.llm = llm
        self.logger = logger
        self._last_score = 0.0  # Track last score for internal use
        self.config = config

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check response against sources for hallucinations"""
        self.logger.info("Checking for hallucinations...")
        try:
            response = state.get("response", "")
            documents = state.get("documents", [])
            web_results = state.get("web_results", [])

            if len(documents) == 0 and len(web_results) == 0:
                return {"hallucination_risk": "HIGH"}

            # Check response against source documents
            self._last_score = self._check_hallucination(response, documents, web_results)

            # Add risk assessment to state without modifying response
            hallucination_risk = self._get_risk_level(self._last_score)
            confidence_score = self._last_score

            self.logger.info(f"Hallucination risk: {hallucination_risk}, confidence score:{confidence_score}")

            return {
                "hallucination_risk": hallucination_risk,
                "confidence_score": confidence_score
            }

        except Exception as e:
            self.logger.error(f"Error detecting hallucinations: {str(e)}")
            return {"hallucination_risk": "UNKNOWN"}

    def _check_hallucination(self, response: str, documents: List[Document], web_results: List[Dict] = None) -> float:
        """Compare response with source documents for verification"""
        prompt = f"""You are an expert fact-checker responsible for verifying information accuracy. Your task is to analyze the response against source documents and calculate a precise verification score.

        Response to verify:
        {response}

        Source Documents:
        {self._format_documents(documents, web_results)}

        SCORING CRITERIA:

        1. SOURCE ALIGNMENT (40 points)
        - Direct Quote Accuracy (20 points)
            * 20: Perfect match with source content
            * 15: Minor paraphrasing but same meaning
            * 10: Partial match with some discrepancies
            * 5: Minimal alignment with sources
            * 0: No matching content found

        - Temporal Relevance (20 points)
            * 20: Information is current and up-to-date
            * 15: Slightly outdated but mostly relevant
            * 10: Significantly outdated information
            * 5: Obsolete information
            * 0: Temporal context completely wrong

        2. FACTUAL ACCURACY (30 points)
        - Technical Precision (15 points)
            * 15: All technical details are correct
            * 10: Minor technical inaccuracies
            * 5: Major technical errors
            * 0: Completely incorrect technical information

        - Numerical Accuracy (15 points)
            * 15: All numbers/versions/dates match sources
            * 10: Minor numerical discrepancies
            * 5: Major numerical errors
            * 0: All numbers are wrong

        3. RESPONSE INTEGRITY (20 points)
        - Logical Consistency (10 points)
            * 10: Perfectly coherent with no contradictions
            * 5: Minor inconsistencies
            * 0: Major logical contradictions

        - Scope Adherence (10 points)
            * 10: Stays within scope of sources
            * 5: Minor scope creep
            * 0: Significant unsupported claims

        4. CONTEXT RELIABILITY (10 points)
        - Context Preservation (10 points)
            * 10: Perfect context maintenance
            * 5: Slight context distortion
            * 0: Complete context loss

        CALCULATION INSTRUCTIONS:
        1. Score each subcategory according to the criteria
        2. Sum all points
        3. Divide by 100 to get final score between 0.0 and 1.0

        RISK LEVELS:
        - 0.0-0.5: HIGH RISK (Major hallucination)
        - 0.51-0.75: MEDIUM RISK (Partial hallucination)
        - 0.76-1.0: LOW RISK (Reliable)

        IMPORTANT: Return ONLY the final decimal score between 0.0 and 1.0.
        Example outputs: 0.95, 0.82, 0.67, 0.43

        Your calculated score with all categories summed:"""

        try:
            response_text = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            # Extract just the number if there's any other text
            import re
            if match := re.search(r'(\d+\.\d+)', response_text):
                score = float(match.group(1))
            else:
                score = float(response_text)
            
            self.logger.info(f"Hallucination score: {score}")
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            self.logger.error(f"Failed to parse hallucination score: {str(e)}")
            return 0.0

    def _format_documents(self, documents: List[Document], web_results: List[Dict] = None) -> str:
        """Format documents for verification"""
        if web_results:  #todo: customize to support the search tool used
            return "\n\n".join([f"Document {i + 1}:\n{result['content']}"
                                for i, result in enumerate(web_results)])
        return "\n\n".join([f"Document {i + 1}:\n{doc.page_content}"
                            for i, doc in enumerate(documents)])

    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level with balanced thresholds"""
        if score <= self.config.get_query_config("hallucination.high_risk", 0.6):
            return "HIGH"
        elif score < self.config.get_query_config("hallucination.medium_risk", 0.8):
            return "MEDIUM"
        else:
            return "LOW"
