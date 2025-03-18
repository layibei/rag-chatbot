from typing import Dict, Any, Literal
from langchain_core.language_models import BaseChatModel
from config.common_settings import CommonConfig
from utils.logging_util import logger

OutputFormat = Literal["code", "table", "markdown"]

class ResponseFormatter:
    """
    Clean response formatter that produces UI-friendly markdown output.
    Supports code blocks and tables while maintaining consistent formatting.
    """

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.config = config
        self.logger = logger
        
        # Format indicators in user queries
        self.format_indicators = {
            "code": [
                "show me the code",
                "write code",
                "code example",
                "implementation",
                "function",
                "class",
                "script"
            ],
            "table": [
                "in table format",
                "as a table",
                "show table",
                "create table",
                "tabular form",
                "comparison table"
            ]
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format response based on user intent and content type"""
        try:
            response = state.get("response", "")
            user_query = state.get("rewritten_query", state.get("user_input", "")).lower()
            
            if not response:
                return state

            # Detect format from user query
            output_format = self._detect_format_from_query(user_query)
            
            # Format response accordingly
            formatted_response = self._format_response(
                response=response,
                format_type=output_format,
                sources=state.get("sources", []),
                metadata=state.get("metadata", {})
            )

            return {
                "response": formatted_response,
                "output_format": output_format
            }

        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return {"response": response, "output_format": "markdown"}

    def _detect_format_from_query(self, query: str) -> OutputFormat:
        """Detect desired format from user query, default to markdown"""
        # query = query.lower()
        
        # Check for explicit format indicators
        # for format_type, indicators in self.format_indicators.items():
        #     if any(indicator in query for indicator in indicators):
        #         self.logger.debug(f"Detected {format_type} format from query: {query}")
        #         return format_type

        # Default to markdown
        return "markdown"

    def _format_response(
        self, 
        response: str, 
        format_type: OutputFormat,
        sources: list = None,
        metadata: dict = None
    ) -> str:
        """Format response based on detected type"""
        
        # Handle specific formats
        if format_type == "code":
            return self._format_code_response(response)
        elif format_type == "table":
            return self._format_table_response(response)
            
        # Default markdown formatting
        return self._format_markdown_response(response, sources, metadata)

    def _format_code_response(self, response: str) -> str:
        """Format code response with proper language highlighting"""
        # If already wrapped in code blocks, return as is
        if response.strip().startswith("```") and response.strip().endswith("```"):
            return response
            
        # Detect language based on content
        first_line = response.split('\n')[0].strip().lower()
        lang = self._detect_language(first_line, response)
        
        # Wrap in code blocks with detected language
        return f"```{lang}\n{response.strip()}\n```"

    def _detect_language(self, first_line: str, content: str) -> str:
        """Detect programming language based on content"""
        # Common language indicators
        indicators = {
            "python": ["def ", "class ", "import ", "from ", "print("],
            "javascript": ["function", "const ", "let ", "var ", "console."],
            "java": ["public class", "private ", "void ", "System.out"],
            "sql": ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE"],
            "html": ["<html", "<div", "<body", "<script", "<style"],
            "css": ["{", "body {", ".class", "#id", "@media"],
            "json": ["{", "[", "\":", "null", "true", "false"],
            "yaml": ["apiVersion:", "kind:", "metadata:", "spec:", "---"]
        }
        
        content_lower = content.lower()
        
        for lang, patterns in indicators.items():
            if any(pattern.lower() in content_lower for pattern in patterns):
                return lang
                
        return ""  # Empty string for unknown language

    def _format_table_response(self, response: str) -> str:
        """Format table response in markdown table format"""
        # If already in markdown table format, return as is
        if "|" in response and "-|-" in response:
            return response
            
        # Convert to markdown table
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            return response
            
        # Try to split by common separators
        for separator in ["|", "\t", ","]:
            if separator in lines[0]:
                return self._convert_to_markdown_table(lines, separator)
                
        return response

    def _convert_to_markdown_table(self, lines: list, separator: str) -> str:
        """Convert separated lines to markdown table format"""
        try:
            # Process headers
            headers = [cell.strip() for cell in lines[0].split(separator)]
            
            # Create table lines
            table_lines = []
            
            # Add header row
            table_lines.append("| " + " | ".join(headers) + " |")
            
            # Add separator row
            table_lines.append("| " + " | ".join("---" for _ in headers) + " |")
            
            # Add data rows
            for line in lines[1:]:
                cells = [cell.strip() for cell in line.split(separator)]
                # Ensure same number of cells as headers
                while len(cells) < len(headers):
                    cells.append("")
                cells = cells[:len(headers)]  # Truncate if too many cells
                table_lines.append("| " + " | ".join(cells) + " |")
                
            return "\n".join(table_lines)
            
        except Exception as e:
            self.logger.error(f"Error converting to markdown table: {str(e)}")
            return response

    def _format_markdown_response(self, response: str, sources: list = None, metadata: dict = None) -> str:
        """Format as markdown with optional sources and metadata"""
        formatted = response.strip()
        
        # Add sources if available
        if sources:
            formatted += "\n\n### Sources\n"
            for idx, source in enumerate(sources, 1):
                title = source.get("title", f"Document {idx}")
                url = source.get("url", "")
                formatted += f"{idx}. [{title}]({url})\n"

        # Add metadata if enabled
        if metadata and self.config.get_query_config("output.include_metadata", False):
            formatted += "\n\n---\n"
            if "response_time" in metadata:
                formatted += f"*Response generated in {metadata['response_time']}s*"

        return formatted
