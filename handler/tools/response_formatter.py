from typing import Dict, Any, List, Optional, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from config.common_settings import CommonConfig
from utils.logging_util import logger
import json

OutputFormat = Literal["chart", "table", "markdown", "code", "json"]


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

            if not response:
                return state

            output_format = self._detect_output_format(response, user_input)
            
            formatters = {
                "json": self._format_json,
                "chart": self._format_chart,
                "table": self._format_table,
                "code": self._format_code,
                "markdown": self._format_markdown
            }
            
            formatted_response = formatters[output_format](response, documents)

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

    def _detect_output_format(self, response: str, user_input: str) -> OutputFormat:
        """Use LLM to intelligently detect the most appropriate output format"""
        prompt = """You are an expert in content analysis and formatting. Analyze this content and determine the optimal output format.

        Content to Analyze:
        User Query: {user_input}
        Response Content: {response}

        Available Output Formats:

        1. JSON Format (Priority: Highest for Data)
           ✓ Structured data that needs to be parsed
           ✓ API-like responses or configurations
           ✓ Nested data structures
           ✓ Machine-readable content
           Example queries: "List all configuration options", "Get user properties"

        2. TABLE Format (Priority: High for Comparisons)
           ✓ Structured comparisons across items
           ✓ Property listings
           ✓ Feature matrices
           ✓ Specification lists
           Example queries: "Compare framework features", "List API endpoints"

        3. CHART Format (Priority: High for Trends)
           ✓ Numerical trends or patterns
           ✓ Statistical distributions
           ✓ Performance metrics
           ✓ Time-series data
           Example queries: "Show performance trends", "Display usage statistics"

        4. CODE Format (Priority: High for Technical)
           ✓ Programming examples
           ✓ Command-line instructions
           ✓ Configuration snippets
           ✓ Technical implementations
           Example queries: "How to implement X", "Show me the code for Y"

        5. MARKDOWN Format (Priority: Default)
           ✓ Explanatory content
           ✓ Mixed content types
           ✓ Documentation
           ✓ Step-by-step guides
           Example queries: "Explain how X works", "What is the process for Y"

        Decision Process:
        1. Analyze the user's intent first
        2. Examine the content structure
        3. Consider data parsability
        4. Evaluate visualization needs
        5. Check for technical requirements

        Return ONLY one word from: json, table, chart, code, markdown

        Your format choice:"""

        try:
            format_response = self.llm.invoke(
                [HumanMessage(content=prompt.format(
                    user_input=user_input,
                    response=response[:1000]
                ))]
            ).content.strip().lower()

            if format_response in ["json", "chart", "table", "markdown", "code"]:
                logger.debug(f"Detected format: {format_response}")
                return format_response
            
            logger.warning(f"Invalid format detected: {format_response}, defaulting to markdown")
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

    def _format_table(self, response: str, documents: List[Document]) -> str:
        """Format response as a clean table"""
        prompt = """Convert the following data into a well-structured table.
        Use ASCII table format for compatibility.
        
        DATA: {response}
        
        Guidelines:
        1. Create clear column headers
        2. Align data properly
        3. Use appropriate separators
        4. Maintain data relationships
        5. Optimize for readability
        
        Format as a table:"""

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt.format(response=response))]).content
            return f"```\n{formatted}\n```"
        except Exception as e:
            logger.error(f"Error formatting table: {str(e)}")
            return response

    def _format_code(self, response: str) -> str:
        """Format response focusing on code blocks"""
        # First try to format with language detection
        formatted = self._format_code_blocks(response)

        # Add code block markers if not present
        if not formatted.strip().startswith("```"):
            formatted = f"```\n{formatted}\n```"

        return formatted

    # def _add_sources(self, response: str, documents: List[Document], format_type: OutputFormat) -> str:
    #     """Add source citations based on format type"""
    #     sources = []
    #     for i, doc in enumerate(documents, 1):
    #         source = doc.metadata.get('source', 'Unknown source')
    #         page = doc.metadata.get('page', '')
    #         citation = f"[{i}] {source}" + (f" (page {page})" if page else "")
    #         sources.append(citation)
    #
    #     if not sources:
    #         return response
    #
    #     if format_type == "markdown":
    #         return response + "\n\n---\n**Sources:**\n" + "\n".join(sources)
    #     elif format_type == "chat":
    #         return response + "\n\nℹ️ Sources:\n" + "\n".join(sources)
    #     elif format_type in ["table", "code"]:
    #         return response + "\n\nSources:\n" + "\n".join(sources)
    #     else:
    #         return response + "\n\nSources:\n" + "\n".join(sources)

    def _format_markdown(self, response: str, documents: List[Document]) -> str:
        """Format the response with proper markdown syntax"""
        prompt = self._create_formatting_prompt(response, documents)

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt)]).content
            return formatted
        except Exception as e:
            logger.error(f"Error in markdown formatting: {str(e)}")
            return response

    def _create_formatting_prompt(self, response: str, documents: List[Document]) -> str:
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

    def _format_json(self, response: str, documents: List[Document]) -> str:
        """Format response as structured JSON"""
        prompt = """Convert the following content into well-structured JSON format.

        Content to Convert:
        {response}

        Guidelines:
        1. Ensure valid JSON syntax
        2. Use appropriate data types:
           - Strings for text
           - Numbers for numerical values
           - Boolean for true/false
           - Arrays for lists
           - Objects for structured data
        3. Use meaningful key names
        4. Maintain data relationships
        5. Include all relevant information
        6. Format nested structures clearly
        7. Preserve numerical precision
        8. Handle special characters properly

        Bad Example:
        {{ "data": "everything in one string" }}

        Good Example:
        {{
            "title": "API Configuration",
            "version": 2.1,
            "settings": {{
                "enabled": true,
                "maxRetries": 3,
                "endpoints": [
                    {{
                        "name": "primary",
                        "url": "https://api.example.com",
                        "timeout": 30
                    }}
                ]
            }}
        }}

        Convert to JSON:"""

        try:
            formatted = self.llm.invoke([HumanMessage(content=prompt.format(response=response))]).content
            # Validate JSON
            json.loads(formatted)  # Will raise JSONDecodeError if invalid
            return f"```json\n{formatted}\n```"
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON generated: {str(e)}")
            return response
        except Exception as e:
            logger.error(f"Error formatting JSON: {str(e)}")
            return response
