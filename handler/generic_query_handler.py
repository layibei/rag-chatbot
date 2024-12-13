from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from handler.workflow.query_process_workflow import QueryProcessWorkflow
from utils.logging_util import logger


class QueryHandler:
    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.logger = logger
        self.llm = llm
        self.vectorstore = vectorstore

    def handle(self, user_input: str):
        # add user intent understand and route to workflow or reply back directly.
        self.logger.info(f"User input: {user_input}")

        # write a prompt to let chatbot know what to do
         # Define the prompt template
        prompt_template = """
        You are a helpful assistant. Your task is to determine if the user input is a greeting. 
        If it is a greeting, respond with a friendly greeting. 
        If it is not a greeting, pass the input to the query process workflow for further handling.

        User Input: {user_input}

        If the input is a greeting, respond with a friendly greeting. 
        If it is not a greeting, return the string "NOT_GREETING".
        """

        # Generate the prompt
        prompt = prompt_template.format(user_input=user_input)

        # Use the language model to determine the intent
        response = self.llm(prompt)

        # Check the response to determine if it's a greeting
        if "NOT_GREETING" in response:
            # Call the query process workflow to handle the input
            return QueryProcessWorkflow(self.llm, self.vectorstore).invoke(user_input)
        else:
            # Return the greeting response
            return response

