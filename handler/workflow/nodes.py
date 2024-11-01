from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from handler.workflow import RequestState
from utils.logging_util import logger
from FlagEmbedding import FlagReranker


class ProcessNodes():
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.logger = logger
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

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

        source = self.question_router.invoke({"user_input": state["user_input"]})
        if source['datasource'] == "vectorstore":
            self.logger.info(f"Routing to vectorstore")
            state["source"] = "vectorstore"
        else:
            self.logger.info(f"Routing to websearch")
            state["source"] = "websearch"

        return state

    def retrieve_documents(self, state: RequestState):
        """
        Retrieve relevant documents from vectorstore.
        :param state:
        :return:
        """
        self.logger.info(f"Retrieving documents")
        question = state["user_input"]
        # documents = self.retriever.get_relevant_documents(question)
        documents = self.retriever.invoke(question)
        self.logger.info(f"Retrieved {len(documents)} documents")
        state['documents'] = documents

        return state

    def grade_documents(self, state: RequestState):
        """
        Grade the relevance of the retrieved documents.
        :param state:
        :return:
        """
        self.logger.info(f"Grading documents")
        question = state["user_input"]
        documents = state['documents']

        self.logger.info(f"Grading {len(documents)} documents")
        filtered_documents = []
        if len(documents) > 0:
            self.logger.info(f"Reranking documents")
            scores = self.reranker.compute_score([(question, doc.page_content) for doc in documents])
            doc_score_pairs = list(zip(documents, scores))
            filtered_documents = self.filter_relevant_documents(doc_score_pairs, question, filtered_documents)
            state['documents'] = filtered_documents

        return state

    def filter_relevant_documents(doc_score_pairs, question: str, filtered_documents: list[Document]) -> list[Document]:
        try:
            # Sort documents by score in descending order
            sorted_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

            # Define a small tolerance for floating-point comparison
            tolerance = 1e-6

            # Filter documents using list comprehension
            relevant_docs = [
                doc for doc in sorted_docs
                if doc[1] >= 1.0 - tolerance
            ]

            # Print results
            for doc in sorted_docs:
                relevance = "relevant" if doc[1] >= 1.0 - tolerance else "not relevant"
                print(f"-- Grade document, result is {relevance}, question:{question}, document:{doc}")

            # Append relevant documents to the filtered list
            filtered_documents.extend(relevant_docs)

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
        question = state.get("user_input")
        documents = state.get("documents", [])

        # Validate input
        if not question:
            raise ValueError("User input is required.")
        if not documents:
            raise ValueError("No documents provided.")

        # Concatenate all documents to form the context
        context = "\n".join(documents)

        template = """
                      You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                      If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                      
                      Question: {user_input}
                      Context: {context}
                      Answer:
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
        state['response'] = generated_result

        return state

    def grade_generation(self, state: RequestState):
        """
        Grade the quality of the generated answer.
        :param state:
        :return:
        """
        self.logger.info(f"Grading generation")
        question = state["user_input"]
        response = state["response"]
        documents = state['documents']

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
        isOk = hallucination_grader.invoke({"documents": "\n".join(documents), "generation": response})
        self.logger.info(f"Grade generation: {isOk}")

        # answer checking
        # answer_prompt = PromptTemplate(
        #     template="""
        #             You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' and 'no'
        #             to indicate whether the answer is useful to resolve a question. Provide the binary score as JSON with a single key
        #             'score' and no preamble or explanation.
        #
        #             Here is the answer: {generation} \n \n
        #
        #             Here is the question: {question} """,
        #     input_variables=['generation', 'question']
        # )
        # answer_grader = answer_prompt | self.llm | JsonOutputParser()

        grade = isOk['score']
        if grade == "yes":
            self.logger.info("Answer is grounded in supported by a set of facts")
            answer_grader_result = answer_grader.invoke({"question": question, "generation": response})
        else:
            self.logger.info("Answer is not grounded in supported by a set of facts")



