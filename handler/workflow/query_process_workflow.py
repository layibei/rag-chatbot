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
from handler.tools.query_rewriter import QueryRewriter
from handler.tools.response_formatter import ResponseFormatter
from handler.tools.response_grader import ResponseGrader
from handler.tools.web_search_tool import WebSearch
from handler.workflow import RequestState
from utils.logging_util import logger
from prompts.constants import PromptManager, PromptTemplate
from utils.audit_logger import get_audit_logger
from config.database.database_manager import DatabaseManager


@dataclass
class Citation:
    source: str  # URL or document name
    content: str  # Relevant excerpt
    confidence: float  # Relevance score


@dataclass
class QueryResponse:
    def __init__(self, answer: str, citations: List[str] = None, suggested_questions: List[str] = None,
                 metadata: Dict = None):
        self.answer = answer
        self.citations = citations or []
        self.suggested_questions = suggested_questions or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response object to a dictionary"""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "suggested_questions": self.suggested_questions,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        """String representation of the response"""
        return str(self.to_dict())


class QueryProcessWorkflow:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore, config: CommonConfig):
        self.logger = logger
        self.llm = llm
        self.config = config
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize tools
        self.web_search = WebSearch(config)
        self.doc_retriever = DocumentRetriever(llm, vectorstore, config)
        self.response_formatter = ResponseFormatter(llm, config)
        self.query_rewriter = QueryRewriter(llm)
        self.response_grader = ResponseGrader(llm, config)
        
        self.graph = self._setup_graph()
        self.max_retries = config.get_query_config("search.max_retries", 1)
        self.fallback_response = "Sorry, i dont have sufficient information to answer your question."
        
        # 初始化审计日志记录器
        db_manager = DatabaseManager(config.get_postgres_uri())
        self.audit_logger = get_audit_logger(db_manager)

    def _setup_graph(self) -> StateGraph:
        memory = MemorySaver()
        workflow = StateGraph(RequestState)

        # Add nodes for each processing step
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("grade_response", self._grade_response)
        workflow.add_node("format_response", self._format_response)
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
        workflow.add_edge("generate_response", "grade_response")

        workflow.add_conditional_edges(
            "grade_response",
            self._should_continue_after_grade_response,
            {
                "rewrite": "rewrite_query",
                "generate_suggested_questions": "generate_suggested_questions"
            }
        )

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

        self.logger.info(f"{len(documents)} documents were found, proceeding to generate response")
        return "generate"

    def _rewrite_query(self, state: RequestState) -> RequestState:
        """Rewrite query for better accuracy"""
        self.logger.info("Rewriting query for better accuracy")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        rewrite_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "rewrite_query", {"original_query": state.get("user_input", "")}
        )
        
        try:
            # 原有的处理逻辑
            rewritten_query = self.query_rewriter.run(state)
            rewrite_attempts = state.get("rewrite_attempts", 0)
            state["rewrite_attempts"] = rewrite_attempts + 1
            state["rewritten_query"] = rewritten_query
            
            # set below as empty
            state["response"] = None
            state["web_results"] = []
            state["documents"] = []
            state["output_format"] = None
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "rewrite_query", rewrite_start, {
                    "rewritten_query": rewritten_query,
                    "attempt_number": rewrite_attempts + 1,
                    "status": "success"
                }
            )
            
        except Exception as e:
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "rewrite_query", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            raise
            
        return state

    def _web_search(self, state: RequestState) -> RequestState:
        """Perform web search"""
        self.logger.info("Performing web search")
        query = state.get("rewritten_query", "")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        web_search_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "web_search", {"query": query}
        )
        
        try:
            # 原有的处理逻辑
            self.logger.debug(f"Web search query: {query}")
            results = self.web_search.run(query)
            state["web_results"] = results
            state["web_search_attempts"] = state.get("web_search_attempts", 0) + 1
            # set it as empty list
            state["documents"] = []
            state["response"] = None
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "web_search", web_search_start, {
                    "result_count": len(results),
                    "status": "success"
                }
            )
            
        except Exception as e:
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "web_search", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            raise
            
        return state

    def _retrieve_documents(self, state: RequestState) -> RequestState:
        """Retrieve relevant documents"""
        self.logger.info("Retrieving relevant documents")
        query = state.get("rewritten_query", "")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        retrieve_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "retrieve_documents", {"query": query}
        )
        
        try:
            # 原有的处理逻辑
            search_config = self.config.get_query_config("search")
            documents = self.doc_retriever.run(query, relevance_threshold=search_config.get("relevance_threshold", 0.7),
                                           max_documents=search_config.get("top_k", 5))
            state["documents"] = documents
            
            # set status as empty
            state["web_results"] = []
            state["response"] = None
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "retrieve_documents", retrieve_start, {
                    "document_count": len(documents),
                    "document_sources": [doc.metadata.get("source", "unknown") for doc in documents[:3]] if documents else [],
                    "status": "success"
                }
            )
            
        except Exception as e:
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "retrieve_documents", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            raise
            
        return state

    def _generate_response(self, state: RequestState) -> RequestState:
        """Generate response strictly based on available sources"""
        query = state.get("rewritten_query", "")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        generate_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "generate_response", {"query": query}
        )
        
        try:
            # 原有的处理逻辑
            documents = state.get("documents", [])
            web_results = state.get("web_results", [])
            
            sources = []
            if documents:
                sources.extend([f"Document: {doc.page_content}" for doc in documents])
            if web_results:
                sources.extend([f"Web Result: {result}" for result in web_results])
            
            self.logger.info(f"Generating response for query: {query}")
            
            if not sources and state.get("rewrite_attempts", 0) >= self.max_retries:
                self.logger.info("No documents found and query rewrite attempts exceeded, returning fallback response")
                state["response"] = self.fallback_response
                state["fallback_response"] = True
                
                # 记录步骤结束（使用回退响应）
                self.audit_logger.end_step(
                    request_id, user_id, session_id, 
                    "generate_response", generate_start, {
                        "used_fallback": True,
                        "status": "success"
                    }
                )
                
                return state
            
            # Get and format prompt using PromptManager
            prompt = self.prompt_manager.format_prompt(
                PromptTemplate.GENERATE_RESPONSE,
                query=query,
                sources="\n\n".join(sources)
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            state["response"] = response
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "generate_response", generate_start, {
                    "response_length": len(response),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response,
                    "used_fallback": False,
                    "status": "success"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            state["response"] = self.fallback_response
            state["fallback_response"] = True
            
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "generate_response", e, {
                    "used_fallback": True,
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            
        return state

    def _grade_response(self, state: RequestState) -> RequestState:
        """Grade response relevance and completeness"""
        self.logger.info("Grading response relevance and completeness")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        grade_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "grade_response", {}
        )
        
        try:
            # 原有的处理逻辑
            if self._is_fallback_response(state):
                self.logger.debug("Fallback response detected, returning empty response")
                
                # 记录步骤结束（使用回退响应）
                self.audit_logger.end_step(
                    request_id, user_id, session_id, 
                    "grade_response", grade_start, {
                        "is_fallback": True,
                        "status": "success"
                    }
                )
                
                return state
            
            score = self.response_grader.run(state)
            state["response_grade_score"] = score
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "grade_response", grade_start, {
                    "score": score,
                    "is_fallback": False,
                    "status": "success"
                }
            )
            
        except Exception as e:
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "grade_response", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            raise
            
        return state

    def _is_fallback_response(self, state: RequestState) -> bool:
        """Check if fallback response is needed"""
        return state.get("fallback_response", False) or state.get("response", "") == self.fallback_response

    def _should_continue_after_grade_response(self, state: RequestState) -> str:
        score = state.get("response_grade_score", 0.0)
        rewrite_attempts = state.get("rewrite_attempts", 0)
        if not self._is_fallback_response(state) and score < self.config.get_query_config("grading.minimum_score",
                                                                                          0.7) and rewrite_attempts < self.max_retries:
            self.logger.debug(
                f"Response grade is below minimum score and rewrite attempts:{rewrite_attempts}, attempting rewrite,"
                f"score:{score}")
            return "rewrite"

        return "generate_suggested_questions"

    def _format_response(self, state: RequestState) -> RequestState:
        """Format the final response"""
        if self._is_fallback_response(state):
            self.logger.debug("Fallback response detected, returning empty response")
            return state

        self.logger.info("Formatting the final response")
        result = self.response_formatter.run(state)

        state['response'] = result.get("response", "")
        state['output_format'] = result.get("output_format", "")
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

        # get query_rewrite_enable from query_config
        query_rewrite_enabled = self.config.get_query_config("search.query_rewrite_enabled", False)
        if not query_rewrite_enabled:
            self.logger.debug(f"Query rewrite is disabled, skipping query rewrite,rewrite_attempts:{rewrite_attempts}")
            return "generate"

        if not documents and not web_results and not rewrite_attempts < self.max_retries and not rewritten_query:
            self.logger.debug(f"No documents are found in retrieval results and web search results, attempting query "
                              f"rewrite,rewrite_attempts:{rewrite_attempts}")
            return "rewrite"

        self.logger.info("Proceeding to generate response")
        return "generate"

    def _generate_suggested_questions(self, state: RequestState) -> RequestState:
        """Generate contextually relevant follow-up questions"""
        self.logger.info("Generating suggested questions")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        suggest_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "generate_suggested_questions", {}
        )
        
        try:
            if not self.config.get_query_config("output.generate_suggested_documents",
                                                False) or self._is_fallback_response(state):
                self.logger.info("Suggested questions generation is disabled")
                return state

            query = state.get("rewritten_query", "")
            response = state.get("response", "")

            prompt = """You are an AI assistant specialized in generating insightful follow-up questions.

            Context:
            Original Query: "{query}"
            Topic Summary: "{summary}"

            Task: Generate 3 follow-up questions that help users explore this topic more deeply.

            Requirements for each question:
            1. Must directly relate to the original query's topic
            2. Should explore different aspects:
               - Technical details or implementation
               - Practical applications or use cases
               - Best practices or common pitfalls
            3. Must be specific and actionable
            4. Must be under 100 characters
            5. Must end with a question mark
            6. Avoid repeating information from the original query
            7. Focus on what's most valuable to understand next

            IMPORTANT: Return ONLY raw JSON without any markdown formatting or code blocks.
            DO NOT include ```json or ``` tags.

            Return in this exact format:
            {{
                "questions": [
                    "First specific follow-up question?",
                    "Second specific follow-up question?",
                    "Third specific follow-up question?"
                ]
            }}

            Examples of good questions:
            - "What are the security implications of implementing this approach?"
            - "How does this compare to [related technology] in terms of performance?"
            - "What are the common pitfalls when scaling this solution?"

            Bad examples:
            - "Can you tell me more?" (too vague)
            - "What is your opinion?" (not specific)
            - "Why is this important?" (too generic)

            Your response (raw JSON only):"""

            result = self.llm.invoke([HumanMessage(content=prompt.format(
                query=query,
                summary=response[:200]
            ))]).content.strip()

            # Remove any markdown code block syntax if present
            result = result.replace('```json', '').replace('```', '').strip()

            try:
                parsed = json.loads(result)
                questions = parsed.get("questions", [])

                if len(questions) == 3 and all(isinstance(q, str) and q.strip().endswith("?") for q in questions):
                    state["suggested_questions"] = questions
                else:
                    self.logger.warning("Invalid questions format or count")
                    state["suggested_questions"] = []

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}\nResponse was: {result}")
                state["suggested_questions"] = []

            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "generate_suggested_questions", suggest_start, {
                    "question_count": len(state.get("suggested_questions", [])),
                    "status": "success"
                }
            )
            
            return state

        except Exception as e:
            self.logger.error(f"Error generating suggested questions: {str(e)}")
            state["suggested_questions"] = []
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "generate_suggested_questions", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            return state

    def _generate_citations(self, state: RequestState) -> RequestState:
        """Generate citations from document and web search results"""
        self.logger.info("Generating citations")
        
        # 记录步骤开始
        request_id = state.get("request_id", "unknown")
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        citation_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "generate_citations", {}
        )
        
        try:
            # need to check if it's enabled or not, if not return state.
            if not self.config.get_query_config("output.generate_citations", False) or self._is_fallback_response(state):
                self.logger.info("Citation generation is disabled")
                return state

            citations = []
            seen_sources = set()  # Track unique sources

            # Process document citations
            documents = state.get("documents", [])
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata.get('source') and doc.metadata.get("source").startswith(('http', 'https')):
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
            
            # 记录步骤结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "generate_citations", citation_start, {
                    "citation_count": len(state.get("citations", [])),
                    "status": "success"
                }
            )
            
            return state

        except Exception as e:
            # 记录错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "generate_citations", e, {
                    "status": "error",
                    "error_type": type(e).__name__
                }
            )
            return state

    def invoke(self, user_input: str, user_id: str, request_id: str, session_id: str, original_query: str) -> Dict[
        str, Any]:
        """
        Invoke the workflow with the given input
        """
        # 记录工作流开始
        workflow_start = self.audit_logger.start_step(
            request_id, user_id, session_id, 
            "query_workflow", {"user_input": user_input, "original_query": original_query}
        )
        
        try:
            # 原有的处理逻辑
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
                "original_query": original_query,
                
                # Initialize fields that will be modified
                "rewritten_query": original_query,
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
                # self.logger.info(s)
                pass
            
            final_state = self.graph.get_state(thread)
            self.logger.debug(f"final response state:{final_state}")
            
            response = QueryResponse(
                answer=final_state.values.get("response", ""),
                citations=final_state.values.get("citations", []),
                suggested_questions=final_state.values.get("suggested_questions", []),
                metadata={"output_format": final_state.values.get("output_format", "")}
            )
            
            # 记录工作流结束
            self.audit_logger.end_step(
                request_id, user_id, session_id, 
                "query_workflow", workflow_start, {
                    "status": "success",
                    "response_summary": {
                        "has_answer": bool(final_state.values.get("response", "")),
                        "has_citations": bool(final_state.values.get("citations", [])),
                        "has_suggested_questions": bool(final_state.values.get("suggested_questions", [])),
                        "answer_length": len(final_state.values.get("response", "")),
                        "citation_count": len(final_state.values.get("citations", [])),
                        "suggested_question_count": len(final_state.values.get("suggested_questions", [])),
                        "output_format": final_state.values.get("output_format", ""),
                        "rewrite_attempts": final_state.values.get("rewrite_attempts", 0),
                        "web_search_attempts": final_state.values.get("web_search_attempts", 0)
                    }
                }
            )
            
            return response.to_dict()  # Convert to dictionary before returning
            
        except Exception as e:
            # 记录工作流错误
            self.audit_logger.error_step(
                request_id, user_id, session_id, 
                "query_workflow", e, {
                    "error_location": "workflow_process",
                    "error_type": type(e).__name__
                }
            )
            self.logger.error(f"Error in query workflow: {str(e)}")
            raise
