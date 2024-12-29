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
        prompt = f"""You are an expert fact-checker responsible for verifying information accuracy.

        TASK: Calculate a verification score by comparing the response against source documents.

        Response to verify:
        {response}

        Source Documents:
        {self._format_documents(documents, web_results)}

        SCORING CALCULATION:

        1. Source Alignment (40 points max):
           - Direct quotes or paraphrasing (20 points)
           - Logical inferences from sources (10 points)
           - Staying within source scope (10 points)
           Score calculation: Count supported statements / total statements * category points

        2. Factual Accuracy (30 points max):
           - Core facts match sources (15 points)
           - Technical terms used correctly (10 points)
           - Numbers and specifics accurate (5 points)
           Score calculation: Number of accurate facts / total facts * category points

        3. Response Integrity (20 points max):
           - No unsupported claims (10 points)
           - Appropriate uncertainty language (5 points)
           - Accurate information synthesis (5 points)
           Score calculation: Deduct points for each violation

        4. Context Reliability (10 points max):
           - Original context preserved (5 points)
           - No contradictions (5 points)
           Score calculation: Full points if preserved, deduct for each issue

        FINAL SCORE CALCULATION:
        1. Calculate points for each category
        2. Sum all category points (max 100)
        3. Divide total by 100 to get final score between 0.0 and 1.0
        4. Round to two decimal places

        Example Calculation:
        Source Alignment:    35/40 points (strong direct support, minor inferences)
        Factual Accuracy:    25/30 points (core facts correct, one number imprecise)
        Response Integrity:  18/20 points (good synthesis, appropriate caveats)
        Context Reliability: 8/10 points (context maintained, minor ambiguity)
        Total: 86/100 = 0.86 final score

        SCORING REFERENCE:
        1.0 (100 points): Perfect alignment with sources
        0.8-0.9 (80-90 points): Strong support with minor imprecisions
        0.6-0.7 (60-70 points): Moderate support with some gaps
        0.0-0.5 (0-50 points): Poor support or major discrepancies

        ANALYSIS STEPS:
        1. Review each statement in response
        2. Check source support for each claim
        3. Calculate points for each category using criteria above
        4. Sum category points
        5. Divide by 100 for final score
        6. Round to two decimal places

        Return ONLY the final decimal score between 0.0 and 1.0.
        Example outputs: 0.95, 0.82, 0.67, 0.43

        Your calculated score with all categories summed:"""

        try:
            score = float(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
            self.logger.info(f"Hallucination score: {score}")
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            self.logger.error(f"Failed to parse hallucination score, {str(e)}")
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
