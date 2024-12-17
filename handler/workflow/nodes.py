from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from datetime import datetime
from pytz import UTC

from handler.workflow import RequestState
from utils.logging_util import logger
from FlagEmbedding import FlagReranker
...
class ProcessNodes():
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.logger = logger
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        self.state = {}
        self.web_search_tool = TavilySearchResults(k=3)
    def _track_step(self, state: RequestState, step_name: str, prompt: str, response: str, tokens: dict):
        """Track a workflow step execution"""
        if "messages" not in state:
            state["messages"] = []
        if "token_usage" not in state:
            state["token_usage"] = {}
            
        state["messages"].append({
            "step": step_name,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now(UTC).isoformat()
        })
        
        # Accumulate token usage
        for key, value in tokens.items():
            state["token_usage"][key] = state["token_usage"].get(key, 0) + value

    def route_query(self, state: RequestState):
        """
        Route user query to corresponding nodes for processing.
        :param state:
        :return: Next node to call
        """
        self.logger.info(f"Routing query to node")
        router_prompt = PromptTemplate(
            template="""
                    You are an expert at routing a user query to a vectorstore or websearch. Use the vectorstore for questions
                    on following topics:
                    - Design patterns for container-based distributed systems.
                    - House price in American cities.
                    - Wheather in China cities.
                    
                    You do not need to be stringent with the keywords in the user query related to these topics. Otherwise, 
                    use web-search tool. 
                    
                    Give a binary choice 'web_search' or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and no preamble or explanation. 
                    Question to route: {user_input} \n
                    """, input_variables=['user_input'],
        )
        self.question_router = router_prompt | self.llm | JsonOutputParser()
        self.state = state;
        source = self.question_router.invoke({"user_input": self.state["user_input"]})
        if source['datasource'] == "vectorstore":
            self.logger.info(f"Routing to vectorstore")
            self.state["route"] = "vectorstore"
        else:
            self.logger.info(f"Routing to websearch")
            self.state["route"] = "websearch"

        self._track_step(
            state=self.state,
            step_name="route_query",
            prompt=router_prompt,
            response=source,
            tokens=self.llm.get_token_usage()
        )

        return self.state['route']
    def retrieve_documents(self, state: RequestState):
        """
        Retrieve relevant documents from vectorstore.
        :param state:
        :return:
        """
        self.logger.info(f"Retrieving documents")
        question = self.state["user_input"]
        # documents = self.retriever.get_relevant_documents(question)
        self.logger.info(f"retrieving documents from vectorstore, user_input:{question}")
        documents = self.vectorstore.as_retriever().invoke(question)
        logger.info(f"Retrieved {len(documents)} documents")
        self.state['documents'] = documents

        return self.state

    def grade_documents(self, state: RequestState):
        """
        Grade the relevance of the retrieved documents.
        :param state:
        :return:
        """
        self.logger.info(f"Grading documents")
        question = self.state["user_input"]
        documents = self.state['documents']

        self.logger.info(f"Grading {len(documents)} documents")
        filtered_documents = []
        if len(documents) > 0:
            self.logger.info(f"Reranking documents")
            scores = self.reranker.compute_score([(question, doc.page_content) for doc in documents])
            doc_score_pairs = list(zip(documents, scores))
            filtered_documents = self.filter_relevant_documents(doc_score_pairs, question)
            self.state['documents'] = filtered_documents

        return self.state
    def filter_relevant_documents(self, doc_score_pairs, question: str) -> list[Document]:
        try:
            # Sort documents by score in descending order
            sorted_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

            # Define a small tolerance for floating-point comparison
            tolerance = 1e-6

            # Filter documents using list comprehension
            relevant_docs = [
                doc[0] for doc in sorted_docs
                if doc[1] >= 1.0 - tolerance
            ]

            # Print results
            for doc in sorted_docs:
                relevance = "relevant" if doc[1] >= 1.0 - tolerance else "not relevant"
                print(f"-- Grade document, result is {relevance}, question:{question}, document:{doc}")

            # Append relevant documents to the filtered list
            return relevant_docs
        except TypeError as e:
            print(f"Error: {e}. Ensure doc_score_pairs is a list of tuples.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []
    def generate(self, state: RequestState):
        """
        Generate answer from the retrieved documents.
        :param state:
        :return:
        """
        # Extract user input and documents from the state
        question = self.state["user_input"]
        documents = self.state["documents"]

        # Validate input
        if not question:
            raise ValueError("User input is required.")
        if not documents:
            raise ValueError("No documents provided.")

        # Concatenate all documents to form the context
        page_contents = [doc.page_content for doc in documents]
        context = "\n".join(page_contents)

        template = """
                      You are an assistant for question-answering tasks. Use the following pieces of retrieved documents to answer the question. 
                      If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                      
                      Question: {user_input} \n
                      Retrieved documents: \n
                      {context} \n\n
                  """
        prompt = PromptTemplate(
            input_variables=["context", "user_input"],
            template=template
        )

        chain = (prompt
                 | self.llm
                 | StrOutputParser())
        # Invoke the chain with the question and context
        generated_result = chain.invoke({"user_input": question, "context": context})
        # Update the state with the generated response
        self.state['response'] = generated_result

        self._track_step(
            state=self.state,
            step_name="generate",
            prompt=prompt,
            response=generated_result,
            tokens=self.llm.get_token_usage()
        )

        return self.state
    def grade_generation(self, state: RequestState):
        """
        Grade the quality of the generated answer.
        :param state:
        :return:
        """
        self.logger.info(f"Grading generation")

        question = self.state["user_input"]
        response = self.state["response"]
        documents = self.state['documents']
        route = self.state['route']

        if not response:
            raise ValueError("No generated response was provided.")

        if route == "vectorstore":
            # fact checking
            hallucination_prompt = PromptTemplate(
                template="""
                        You are a grader assessing whether an answer is grounded in supported by a set of facts. Give a binary score 'yes'
                        or 'no' score to indicate whether the answer is grounded in supported by a set of facts. Provide the binary score
                        as a JSON with a single key 'score' and no preamble or explanation.
                        Here are the facts:
                        {documents} \n\n
                        Here is the answer: {generation} \n
                        """, input_variables=['generation', 'documents'])
            hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()
            page_contents = [doc.page_content for doc in documents]
            context = "\n".join(page_contents)
            isOk = hallucination_grader.invoke({"documents": context, "generation": response})
            self.logger.info(f"Grade generation: {isOk}")
            grade = None
            if isOk is not None:
                grade = isOk['score']
            if grade == "yes":
                self.logger.info("Answer is grounded in supported by a set of facts")
                print("--- Grade generation does not address question---")
                return "successfully"
            else:
                self.logger.info("Answer is not grounded in supported by a set of facts")
                self.state['response'] = "Sorry, I don't know the answer"
                return "failed"
        elif route == "websearch":
            # answer checking
            answer_prompt = PromptTemplate(
                template="""
                        You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' and 'no'
                        to indicate whether the answer is useful to resolve a question. Provide the binary score as JSON with a single key
                        'score' and no preamble or explanation.
    
                        Here is the answer: {generation} \n \n
    
                        Here is the question: {question} """,
                input_variables=['generation', 'question']
            )
            answer_grader = answer_prompt | self.llm | JsonOutputParser()
            answer_grader_result = answer_grader.invoke({"question": question, "generation": response})
            if answer_grader_result["score"] == "yes":
                print("--- Grade generation addresses question---")
                return "successfully"
            print("--- Grade generation does not address question---")
            self.state['response'] = "Sorry, I don't know the answer"
            return "failed"
            grade = isOk['score']
        else:
            self.state['response'] = "Sorry, I don't know the answer"
            return "failed"
    def web_search(self, state: RequestState):
        """
        Determines whether to generate an answer, or add web search.

        Args:
            state (dict): the current graph state
        Returns:
            state (dict): Binary decision for next node.
        """
        question = self.state["user_input"]
        documents = self.state.get("documents", [])
        # web_search_count = self.state.get("web_search_count", 0)

        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d['content'] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]

        # state['web_search_count'] = web_search_count + 1
        self.state["documents"] = documents

        return self.state