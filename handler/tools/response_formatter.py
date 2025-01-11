from typing import Dict, Any, List, Optional, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from config.common_settings import CommonConfig
from utils.logging_util import logger

OutputFormat = Literal["chart", "table", "markdown", "code"]


class ResponseFormatter:
    """
    Formats the final response from the query processing flow based on content type.
    Supports multiple output formats:
    - markdown: For general documentation-style responses
    - chat: For conversational responses
    - table: For structured data
    - code: For code examples and technical responses
    - plain: For simple text responses
    """

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.llm = llm
        self.config = config

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = state.get("response", "")
            documents = state.get("documents", state.get("web_results", []))
            user_input = state.get("rewritten_query", state.get("user_input", ""))
            self.logger.info(f"Formatting the generation:{response}")

            if not response:
                return state

            # Detect format using LLM
            output_format = self._detect_output_format(response, user_input)
            
            # Format based on detected type
            formatted_response = {
                "chart": self._format_chart,
                "table": self._format_table,
                "code": self._format_code,
                "markdown": self._format_markdown,
                "text": self._format_text,
            }[output_format](response)


            return {
                "response": formatted_response,
                "output_format": output_format
            }

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {
                "response": state.get("response", ""),
                "output_format": "markdown"
            }

    def _format_text(self, response: str):
        self.logger.info(f"Detected format: text for response")
        return response.strip()

    def _detect_output_format(self, response: str, user_input: str) -> OutputFormat:
        """Use LLM to intelligently detect the most appropriate output format"""
        prompt = """As a format detection AI expert, analyze this content and determine the optimal output format.
        Choose from: chart, table, markdown, or code.
        
        USER QUERY: {user_input}
        CONTENT TO ANALYZE: {response}

        FORMAT CRITERIA:
        1. CHART Format (Choose if):
           - Contains numerical data that would benefit from visualization
           - Describes trends, comparisons, or distributions
           - Mentions statistics, percentages, or time-series data
           - User asks for visual representation
           Examples: price trends, performance metrics, statistical distributions

        2. TABLE Format (Choose if):
           - Contains structured, tabular data
           - Presents comparisons across multiple attributes
           - Lists properties or specifications
           - Contains row-column structured information
           Examples: product comparisons, feature lists, configuration settings

        3. CODE Format (Choose if):
           - Contains programming syntax
           - Shows command-line instructions
           - Includes technical implementation details
           - Demonstrates software configurations
           Examples: code snippets, API usage, configuration files

        4. MARKDOWN Format (Choose if):
           - Contains explanatory text with structure
           - Requires hierarchical organization
           - Needs formatting for readability
           - Contains mixed content types
           Examples: technical documentation, step-by-step guides
        5. Plain TEXT:
            if None of above is matched, then default as TEXT.

        INSTRUCTIONS:
        1. Analyze both the user query and the content
        2. Consider the primary purpose of the information
        3. Choose the format that best serves the information's purpose
        4. Return ONLY ONE of: chart, table, markdown, code, text
        5. Avoid any content adjustment and only do the formatting

        RETURN FORMAT: Single word response (chart/table/markdown/code/text)"""

        try:
            format_response = self.llm.invoke(
                [HumanMessage(content=prompt.format(
                    user_input=user_input,
                    response=response[:1000]  # Limit content length for token efficiency
                ))]
            ).content.strip().lower()

            # Validate and return format
            if format_response in ["chart", "table", "markdown", "code", "text"]:
                logger.debug(f"Detected format: {format_response} for response")
                return format_response
            
            logger.warning(f"Invalid format detected: {format_response}, falling back to markdown")
            return "markdown"
            
        except Exception as e:
            logger.error(f"Error in format detection: {str(e)}")
            return "markdown"

    def _format_chart(self, response: str, documents: List[Document]) -> str:
        """Format response as a chart visualization"""
        prompt = """Convert the following data into a clear chart representation.
        Use ASCII/Unicode characters to create a visual chart.
        
        DATA: {response}
        
        Guidelines:
        1. Choose appropriate chart type (bar, line, scatter)
        2. Include axis labels and scales
        3. Add title and legend if needed
        4. Ensure readability in monospace font
        5. Keep data accuracy
        
        Format as a chart:"""

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt.format(response=response))]).content
            return f"```chart\n{formatted}\n```"
        except Exception as e:
            logger.error(f"Error formatting chart: {str(e)}")
            return response

    def _format_table(self, response: str) -> str:
        """Format tables in the response for better client-side rendering"""
        
        prompt = """Analyze and convert any tabular data in this response into a structured JSON format.

        RULES:
        1. DETECT TABLE PATTERNS:
           - Markdown tables (|---|---|)
           - Lists that compare items
           - Property-value pairs
           - Structured comparisons
           - Version or feature matrices

        2. TABLE STRUCTURE:
        {
            "type": "table",
            "data": {
                "headers": ["Column1", "Column2"],
                "rows": [
                    ["Value1", "Value2"],
                    ["Value3", "Value4"]
                ],
                "title": "Optional title",
                "description": "Optional description"
            }
        }

        3. FORMATTING RULES:
           - Keep headers concise and clear
           - Ensure data alignment in rows
           - Preserve numerical precision
           - Maintain data relationships
           - Remove redundant information

        4. TEXT HANDLING:
           - Keep non-table text unchanged
           - Preserve text before and after tables
           - Maintain paragraph breaks

        Original response:
        {response}

        Return format:
        - For sections with tables: <table>{json}</table>
        - For regular text: keep as is
        """

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            return formatted
        except Exception as e:
            self.logger.error(f"Table formatting failed: {str(e)}")
            return response

    def _format_code(self, response: str) -> str:
        """Format response focusing on code blocks"""
        # First try to format with language detection
        formatted = self._format_code_blocks(response)

        # Add code block markers if not present
        if not formatted.strip().startswith("```"):
            formatted = f"```\n{formatted}\n```"

        return formatted

    def _format_markdown(self, response: str) -> str:
        """Format the response with proper markdown syntax"""
        prompt = self._create_formatting_prompt(response)

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt)]).content
            return formatted
        except Exception as e:
            logger.error(f"Error in markdown formatting: {str(e)}")
            return response

    def _create_formatting_prompt(self, response: str) -> str:
        """Create a prompt for the LLM to format the response"""
        return f"""Format the following response in clear markdown, including:
            - Proper headings
            - Code blocks with language tags
            - Lists and tables where appropriate
            - Bold/italic for emphasis
            - Proper line breaks

            Response to format:
            {response}

            Guidelines:
            1. Use ```language for code blocks with appropriate language tags
            2. Use proper markdown tables if there's tabular data
            3. Use bullet points for lists
            4. Add bold for important terms
            5. Preserve any technical accuracy while improving readability
            6. Keep all technical information intact

            Format the response while maintaining its technical accuracy:"""

    def _format_code_blocks(self, text: str) -> str:
        """Ensure code blocks have proper language tags"""
        # This is a fallback if LLM formatting fails
        import re
        code_block_pattern = r'```(?![\w+])(.*?)```'

        def add_language_tag(match):
            code = match.group(1)
            # Try to detect language or default to plaintext
            if 'public class' in code or 'import java' in code:
                lang = 'java'
            elif 'def ' in code or 'import ' in code:
                lang = 'python'
            elif '<html' in code:
                lang = 'html'
            else:
                lang = 'plaintext'
            return f'```{lang}{code}```'

        return re.sub(code_block_pattern, add_language_tag, text, flags=re.DOTALL)
