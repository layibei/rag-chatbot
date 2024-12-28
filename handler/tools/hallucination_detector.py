from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from utils.logging_util import logger

class HallucinationDetector:
    """Detects potential hallucinations in LLM responses"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logger
        self._last_score = 0.0  # Track last score for internal use

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check response against sources for hallucinations"""
        try:
            response = state.get("response", "")
            documents = state.get("documents", [])
            
            if not documents:
                return {"hallucination_risk": "HIGH"}
                
            # Check response against source documents
            self._last_score = self._check_hallucination(response, documents)
            
            # Add risk assessment to state without modifying response
            hallucination_risk = self._get_risk_level(self._last_score)
            confidence_score = self._last_score
            
            return {
                "hallucination_risk": hallucination_risk,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting hallucinations: {str(e)}")
            return {"hallucination_risk": "UNKNOWN"}

    def _check_hallucination(self, response: str, documents: List[Document]) -> float:
        """Compare response with source documents for verification"""
        prompt = f"""You are an expert fact-checker with exceptional analytical skills.

        Task: Analyze the response against source documents to determine factual accuracy and information reliability.

        Response to verify:
        {response}

        Source Documents:
        {self._format_documents(documents)}

        Verification Framework:
        1. Content Accuracy (40%):
           - Core Facts: Are the main points directly supported by sources?
           - Details: Are specific details, numbers, and claims accurate?
           - Terminology: Is domain-specific language used correctly?
           - Temporal Context: Are time-sensitive statements accurate?

        2. Source Alignment (30%):
           - Direct Support: What percentage is explicitly supported by sources?
           - Inference Quality: Are logical conclusions well-supported?
           - Information Gaps: Are unsupported claims clearly identified?
           - Context Preservation: Is the original context maintained?

        3. Response Integrity (20%):
           - Scope Adherence: Does it stay within the source material scope?
           - Qualification: Are limitations and uncertainties acknowledged?
           - Balance: Are multiple viewpoints (if present) represented fairly?
           - Synthesis: Is information combined accurately from multiple sources?

        4. Practical Reliability (10%):
           - Actionability: Can the information be reliably acted upon?
           - Completeness: Are essential caveats and prerequisites included?
           - Currency: Is temporal relevance considered appropriately?
           - Verifiability: Can key points be cross-referenced?

        Scoring Guide:
        1.0: Fully verified
           - All information directly supported by sources
           - Perfect accuracy in facts and details
           - Appropriate context and qualifications
           - No unsupported claims

        0.8-0.9: Highly reliable
           - Core information supported by sources
           - Minor details may have slight imprecisions
           - Good context preservation
           - Well-qualified statements

        0.5-0.7: Partially verified
           - Main points somewhat supported
           - Some details lack direct source support
           - Context mostly preserved
           - Some unsupported but plausible inferences

        0.0-0.4: Poorly verified
           - Major claims lack source support
           - Significant factual discrepancies
           - Context distortion
           - Multiple unsupported assertions

        Analyze thoroughly and return only a single decimal number between 0.0 and 1.0:"""

        try:
            score = float(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.0

    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for verification"""
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                           for i, doc in enumerate(documents)])

    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level with balanced thresholds"""
        if score >= 0.80:
            return "LOW"
        elif score >= 0.60:
            return "MEDIUM"
        return "HIGH" 