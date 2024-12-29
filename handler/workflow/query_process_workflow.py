from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass

from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langgraph.constants import END
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config.common_settings import CommonConfig
from handler.tools.document_retriever import DocumentRetriever
from handler.tools.hallucination_detector import HallucinationDetector
from handler.tools.query_rewriter import QueryRewriter
from handler.tools.response_formatter import ResponseFormatter
from handler.tools.web_search_tool import WebSearch
from handler.workflow import RequestState
from utils.logging_util import logger


@dataclass
class Citation:
    source: str  # URL or document name
    content: str  # Relevant excerpt
    confidence: float  # Relevance score


@dataclass
class QueryResponse:
    answer: str
    citations: List[str]
    suggested_questions: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "suggested_questions": self.suggested_questions,
            "metadata": self.metadata
        }


class QueryProcessWorkflow:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        self.logger = logger
        self.llm = llm
        self.config = config

        # Initialize tools
        self.web_search = WebSearch(config)
        self.doc_retriever = DocumentRetriever(llm, vectorstore, config)
        self.response_formatter = ResponseFormatter(llm, config)
        self.query_rewriter = QueryRewriter(llm)
        self.hallucination_detector = HallucinationDetector(llm, config)

        self.graph = self._setup_graph()
        self.max_retries = config.get_query_config("search.max_retries", 1);

    def _setup_graph(self) -> StateGraph:
        memory = MemorySaver()
        workflow = StateGraph(RequestState)

        # Add nodes for each processing step
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("detect_hallucination", self._detect_hallucination)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("enhance_response", self._enhance_response)
        workflow.add_node("generate_suggested_questions", self._generate_suggested_questions)
        workflow.add_node("generate_citations", self._generate_citations)

        # Set document retrieval as entry point
        workflow.set_entry_point("retrieve_documents")

        # Add conditional edges based on document retrieval results
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._should_try_web_search,
            {
                "web_search": "web_search",
                "rewrite": "rewrite_query",
                "generate": "generate_response"
            }
        )
        # Add conditional edges from web search
        workflow.add_conditional_edges(
            "web_search",
            self._should_rewrite_query,
            {
                "rewrite": "rewrite_query",
                "generate": "generate_response"
            }
        )

        # Rewrite query loops back to document retrieval
        workflow.add_edge("rewrite_query", "retrieve_documents")

        # Response generation and verification flow
        workflow.add_edge("generate_response", "detect_hallucination")

        workflow.add_conditional_edges(
            "detect_hallucination",
            self._handle_hallucination,
            {
                "rewrite": "rewrite_query",
                "enhance": "enhance_response",
                "format": "format_response"
            }
        )

        workflow.add_edge("enhance_response", "generate_suggested_questions")
        workflow.add_edge("generate_suggested_questions", "generate_citations")
        workflow.add_edge("generate_citations", "format_response")

        workflow.add_edge("format_response", END)

        return workflow.compile(checkpointer=memory)

    def _should_try_web_search(self, state: RequestState) -> str:
        """Determine if web search should be attempted or query should be rewritten"""
        self.logger.info("Determining if web search should be attempted or query should be rewritten")
        documents = state.get("documents", [])
        rewrite_attempts = state.get("rewrite_attempts", 0.0)

        web_search_enabled = self.config.get_query_config("search.web_search_enabled", False)

        if not documents:
            if web_search_enabled:
                self.logger.debug("No documents found, attempting web search")
                return "web_search"
            elif rewrite_attempts < self.max_retries:
                self.logger.debug("No documents found and web search is disabled, attempting query rewrite")

                return "rewrite"

        self.logger.info("Documents found, proceeding to generate response")
        return "generate"

    def _rewrite_query(self, state: RequestState) -> RequestState:
        """Rewrite query for better accuracy"""
        self.logger.info("Rewriting query for better accuracy")
        rewritten_query = self.query_rewriter.run(state)
        rewrite_attempts = state.get("rewrite_attempts", 0)
        state["rewrite_attempts"] = rewrite_attempts + 1

        state["rewritten_query"] = rewritten_query

        # set below as empty
        state["response"] = None
        state["web_results"] = []
        state["documents"] = []
        state["output_format"] = None

        return state

    def _web_search(self, state: RequestState) -> RequestState:
        """Perform web search"""
        self.logger.info("Performing web search")
        query = state.get("rewritten_query", "")
        self.logger.debug(f"Web search query: {query}")
        results = self.web_search.run(query)
        state["web_results"] = results
        state["web_search_attempts"] = state.get("web_search_attempts", 0) + 1
        # set it as empty list
        state["documents"] = []
        state["response"] = None
        return state

    def _retrieve_documents(self, state: RequestState) -> RequestState:
        """Retrieve relevant documents"""
        self.logger.info("Retrieving relevant documents")
        query = state.get("rewritten_query", "")
        self.logger.debug(f"Retrieving documents: {query}")

        # get search config
        search_config = self.config.get_query_config("search")
        documents = self.doc_retriever.run(query, relevance_threshold=search_config.get("relevance_threshold", 0.7),
                                           max_documents=search_config.get("top_k", 5))
        state["documents"] = documents

        # set status as empty
        state["web_results"] = []
        state["response"] = None

        return state

    def _generate_response(self, state: RequestState) -> RequestState:
        """Generate comprehensive response based on available sources"""
        query = state["rewritten_query"]
        documents = state.get("documents", [])
        web_results = state.get("web_results", [])
        state["response"] = None
        state["output_format"] = ""

        sources = []
        if documents:
            sources.extend([f"Document: {doc.page_content}" for doc in documents])
        if web_results:
            sources.extend([f"Web Result: {result}" for result in web_results])

        prompt = f"""You are a precise and concise assistant. Generate a clear, factual response based on the provided sources.
            
            Query: "{query}"
            
            Sources:
            {chr(10).join(sources)}
            
            Requirements:
            1. Be direct and concise
            2. Use only provided source information
            3. Include specific facts and figures
            4. Maintain response accuracy
            
            Format:
            - Start with direct answer
            - Use bullet points for multiple facts
            - Include code blocks if technical
            - Keep response under 200 words
            
            Response:"""

        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        state["response"] = response
        return state

    def _detect_hallucination(self, state: RequestState) -> RequestState:
        """Check for potential hallucinations"""
        self.logger.info("Checking for potential hallucinations")
        result = self.hallucination_detector.run(state)

        state["confidence_score"] = result.get("confidence_score", 0.0)
        state["hallucination_risk"] = result.get("hallucination_risk", "")

        return state

    def _format_response(self, state: RequestState) -> RequestState:
        """Format the final response"""
        self.logger.info("Formatting the final response")
        result = self.response_formatter.run(state)

        state['response'] = result.get("response", "")
        state['output_format'] = result.get("output_format", "")
        return state

    def _enhance_response(self, state: RequestState) -> RequestState:
        """Enhance response quality for medium/high risk responses"""
        self.logger.info("Enhancing response quality for medium/high risk responses")
        try:
            response = state.get("response", "")
            documents = state.get("documents", state.get("web_result", ""))

            prompt = f"""Enhance this response with source-verified facts. Be concise and precise.

                Original: {response}
                
                Sources:
                {self._format_documents(documents)}
                
                Requirements:
                1. Add specific facts from sources
                2. Keep original key points
                3. Stay under 200 words
                4. Use markdown formatting
                5. Maintain technical accuracy
                
                Enhanced response:"""

            enhanced = self.llm.invoke([HumanMessage(content=prompt)]).content
            state["response"] = enhanced
            return state
        except Exception as e:
            self.logger.error(f"Error enhancing response: {str(e)}")
            return state

    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for verification"""
        return "\n\n".join([f"Document {i + 1}:\n{doc.page_content}"
                            for i, doc in enumerate(documents)])

    def _should_rewrite_query(self, state: RequestState) -> str:
        """Determine if query needs rewriting based on results"""
        self.logger.info("Determining if query needs rewriting")
        documents = state.get("documents", [])
        web_results = state.get("web_results", [])
        rewrite_attempts = state.get("rewrite_attempts", 0)
        rewritten_query = state.get("rewritten_query", "")

        if not documents and not web_results and rewrite_attempts < self.max_retries and not rewritten_query:
            self.logger.debug(f"No results found, attempting query rewrite,rewrite_attempts:{rewrite_attempts}")
            return "rewrite"

        self.logger.info("Proceeding to generate response")
        return "generate"

    def _handle_hallucination(self, state: RequestState) -> str:
        """Handle detected hallucinations"""
        self.logger.info("Handling potential hallucination")

        if state.get("hallucination_risk") in ["MEDIUM", "HIGH"]:
            rewrite_attempts = state.get("rewrite_attempts", 0)
            if rewrite_attempts >= self.max_retries:
                self.logger.debug(
                    f"Maximum rewrites reached, attempting enhancement,rewrite_attempts:{rewrite_attempts}")
                return "enhance"
            self.logger.debug(f"Attempting query rewrite due to hallucination risk,rewrite_attempts:{rewrite_attempts}")
            return "rewrite"

        return "format"

    def _generate_suggested_questions(self, state: RequestState) -> RequestState:
        """Generate contextually relevant follow-up questions"""
        self.logger.info("Generating suggested questions")
        try:
            # check if enabled
            if not self.config.get_query_config("output.generate_suggested_documents", False):
                self.logger.info("Suggested questions generation is disabled")
                return state

            query = state.get("rewritten_query", "")
            response = state.get("response", "")
            documents = state.get("documents", state.get("web_results", []))

            prompt = f"""Generate 3 follow-up questions based on the user's query.

                User Query: "{query}"
                Current Response: "{response[:200]}..."
                Context: {self._format_documents(documents)}

                Requirements:
                - Each question should be under 100 characters
                - Questions must be specific
                - You could focus on deeper aspects of the original query
                - No general or basic questions
                - No explanations or answers

                Return ONLY an array of 3 questions:
                [
                    "How does X specific implementation work?",
                    "What are the performance implications of Y?",
                    "Which technical constraints affect Z?"
                ]"""

            result = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Clean and parse response
            content = result.content.strip()
            # Remove any markdown formatting if present
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                questions = json.loads(content)
                if isinstance(questions, list):
                    state["suggested_questions"] = questions[:3]
                else:
                    self.logger.warning("Invalid response format: not a list")
                    state["suggested_questions"] = []
                return state

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}")
                state["suggested_questions"] = []
                return state

        except Exception as e:
            self.logger.error(f"Error generating suggested questions: {str(e)}")
            state["suggested_questions"] = []
            return state

    def _generate_citations(self, state: RequestState) -> RequestState:
        """Generate citations from document and web search results"""
        self.logger.info("Generating citations")
        citations = []
        seen_sources = set()  # Track unique sources

        # Process document citations
        documents = state.get("documents", [])
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                source = doc.metadata['source']
                # Skip if we've already seen this source
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                citations.append(Citation(
                    source=source,
                    content="",  # First 200 chars as excerpt
                    confidence=doc.metadata.get('score', 0.0)
                ))

        # Process web search citations
        web_results = state.get("web_results", [])
        for result in web_results:
            if isinstance(result, dict) and result.get('url'):
                source = result['url']
                # Skip if we've already seen this source
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                citations.append(Citation(
                    source=source,
                    content="",  # Empty content since we don't need it
                    confidence=result.get('relevance_score', 0.0)
                ))

        # Sort citations by confidence score
        citations.sort(key=lambda x: x.confidence, reverse=True)
        state['citations'] = [x.source for x in citations[:3]]

        return state

    def invoke(self, user_input: str, user_id: str, request_id: str, session_id: str) -> Dict[str, Any]:
        """
        Invoke the workflow with the given input
        """
        thread = {
            'configurable': {'thread_id': 1}
        }

        # Initialize complete state with all fields that will be modified
        initial_state = {
            # Required fields
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "user_input": user_input,

            # Initialize fields that will be modified
            "rewritten_query": user_input,
            "documents": [],
            "web_results": [],
            "response": None,
            "hallucination_risk": None,
            "confidence_score": 0.0,
            "output_format": "",
            "messages": [],

            # Initialize counters
            "rewrite_attempts": 0,
            "web_search_attempts": 0,
            "enhance_attempts": 0
        }

        for s in self.graph.stream(initial_state, thread):
            self.logger.info(s)

        final_state = self.graph.get_state(thread)
        self.logger.debug(f"final response state:{final_state}")

        response = QueryResponse(
            answer=final_state.values.get("response", ""),
            citations=final_state.values.get("citations", []),
            suggested_questions=final_state.values.get("suggested_questions", []),
            metadata={"output_format": final_state.values.get("output_format", "")}
        )

        return response.to_dict()  # Convert to dictionary before returning
