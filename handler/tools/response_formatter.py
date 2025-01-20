from typing import Dict, Any, List, Optional, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from config.common_settings import CommonConfig
from utils.logging_util import logger
import re
from typing import List, Tuple, Dict

OutputFormat = Literal["chart", "table", "markdown", "code", "text"]


class ResponseFormatter:
    """
    Formats the final response with minimal modifications while ensuring consistent structure.
    Supports multiple output formats with pattern-based detection and validation.
    """

    def __init__(self, llm: BaseChatModel, config: CommonConfig):
        self.llm = llm
        self.config = config
        self.logger = logger
        
        # Fixed patterns for format detection with corrected regex
        self.format_patterns = {
            "code": [
                r"```[\w+].*?```",  # Code blocks with language
                r"(?:^|\n)(?:import|from|def|class|public|function)"  # Code keywords
            ],
            "table": [
                r"\|[^|]+\|[^|]+\|",          # Basic table structure
                r"\|[\s\-:]+\|[\s\-:]+\|",    # Fixed table header separator
                r"[+][\-=]+[+][\-=]+[+]",     # Fixed ASCII table borders
                r"(?:\w+,){2,}\w+"            # CSV-like data
            ],
            "chart": [
                r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b.*?(?:\d+(?:\.\d+)?)",  # Time series
                r"\b\d+(?:\.\d+)?%",                                      # Percentages
                r"\b\d+(?:\.\d+)?(?:k|m|b|t|K|M|B|T)?\b",               # Quantities
                r"\d+(?:\.\d+)?\s*(?:vs|versus|compared to)\s*\d+(?:\.\d+)?"  # Comparisons
            ]
        }
        
        # Language detection patterns
        self.code_language_patterns = {
            "python": [r"import \w+", r"def \w+", r"class \w+", r"print\("],
            "java": [r"public class", r"private \w+", r"System\.", r"void \w+"],
            "javascript": [r"const \w+", r"function \w+", r"let \w+", r"console\."],
            "sql": [r"SELECT", r"INSERT INTO", r"CREATE TABLE", r"UPDATE \w+"],
            "html": [r"<\w+>", r"</\w+>", r"<html", r"<div"],
            "shell": [r"#!/bin/\w+", r"\$ \w+", r"echo ", r"sudo "]
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for response formatting"""
        try:
            response = state.get("response", "")
            user_input = state.get("rewritten_query", state.get("user_input", ""))
            
            if not response:
                return state

            # 1. Initial format detection using patterns
            detected_format = self._detect_format_patterns(response)
            
            # 2. LLM confirmation only if pattern detection is ambiguous
            if not detected_format:
                detected_format = self._confirm_format_llm(response, user_input)

            # 3. Apply formatting with validation
            formatted_response = self._apply_format(response, detected_format)
            
            # 4. Validate output structure
            if not self._validate_formatted_output(formatted_response, detected_format):
                self.logger.warning("Format validation failed, using plain text")
                return {
                    "response": response.strip(),
                    "output_format": "text"
                }

            return {
                "response": formatted_response,
                "output_format": detected_format
            }

        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return {"response": response, "output_format": "text"}

    def _detect_format_patterns(self, response: str) -> str:
        """Detect format using regex patterns first"""
        for format_type, patterns in self.format_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response, re.MULTILINE):
                    self.logger.debug(f"Detected {format_type} format using pattern")
                    return format_type
                    
        return "text" if len(response.split()) < 30 else "markdown"

    def _confirm_format_llm(self, response: str, user_input: str) -> str:
        """Use LLM only to confirm format in ambiguous cases"""
        prompt = """Analyze ONLY the structure and return ONE format type.
        Options: chart, table, code, markdown, text
        
        Content: {response}
        
        Rules:
        1. chart: Only for actual numerical data/trends
        2. table: Only for actual tabular data
        3. code: Only for actual code/commands
        4. text: For simple, short responses
        5. markdown: For everything else
        
        Return ONLY ONE WORD."""

        try:
            format_result = self.llm.invoke([
                HumanMessage(content=prompt.format(response=response))
            ]).content.lower().strip()
            
            return format_result if format_result in ["chart", "table", "code", "markdown", "text"] else "text"
        except:
            return "text"

    def _validate_formatted_output(self, formatted: str, format_type: str) -> bool:
        """Validate the formatted output matches expected structure"""
        if format_type == "code" and "```" not in formatted:
            return False
        if format_type == "table" and "|" not in formatted:
            return False
        if format_type == "chart" and "```chart" not in formatted:
            return False
        return True

    def _apply_format(self, response: str, format_type: str) -> str:
        """Apply formatting with minimal content modification"""
        if format_type == "text":
            return response.strip()
            
        if format_type == "code":
            return self._format_code_minimal(response)
            
        if format_type == "table":
            return self._format_table_minimal(response)
            
        if format_type == "chart":
            return self._format_chart_minimal(response)
            
        # Default markdown formatting
        return self._format_markdown_minimal(response)

    def _format_code_minimal(self, response: str) -> str:
        """Format code blocks with language detection"""
        # Check if already properly formatted
        if re.match(r"^```\w*\n.*?\n```$", response, re.DOTALL):
            return response
            
        # Extract existing code blocks or treat entire response as code
        code_block_pattern = r"```(.*?)```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if not code_blocks:
            # Treat entire response as code if it looks like code
            if any(re.search(pattern, response) for patterns in self.code_language_patterns.values() for pattern in patterns):
                code_blocks = [response]
            else:
                return f"```\n{response.strip()}\n```"
        
        formatted_blocks = []
        for block in code_blocks:
            # Detect language
            lang = "plaintext"
            for language, patterns in self.code_language_patterns.items():
                if any(re.search(pattern, block, re.IGNORECASE) for pattern in patterns):
                    lang = language
                    break
            
            # Format block with detected language
            formatted_block = f"```{lang}\n{block.strip()}\n```"
            formatted_blocks.append(formatted_block)
        
        return "\n\n".join(formatted_blocks)

    def _format_table_minimal(self, response: str) -> str:
        """Format tables with consistent structure and alignment"""
        def parse_table_data(text: str) -> Tuple[List[str], List[List[str]]]:
            """Extract headers and rows from various table formats"""
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Handle existing markdown tables
            if re.match(self.format_patterns["table"][0], text):
                rows = [
                    [cell.strip() for cell in line.strip('|').split('|')]
                    for line in lines if not re.match(r'\|[\s-:|]+\|', line)
                ]
                return rows[0], rows[1:]
            
            # Handle ASCII tables
            if re.match(self.format_patterns["table"][1], text):
                data_lines = [line for line in lines if not re.match(r'[+\|][-+=]+[+\|]', line)]
                rows = [
                    [cell.strip() for cell in re.split(r'\s{2,}|\|', line.strip('|'))]
                    for line in data_lines
                ]
                return rows[0], rows[1:]
            
            # Handle CSV-like data
            if re.match(self.format_patterns["table"][2], text):
                rows = [line.split(',') for line in lines]
                return rows[0], rows[1:]
            
            # Handle space-aligned columns
            if re.match(self.format_patterns["table"][3], text):
                sample = lines[0]
                spaces = [i for i, char in enumerate(sample) if char == ' ' and sample[i-1] != ' ']
                if spaces:
                    rows = []
                    for line in lines:
                        row = []
                        start = 0
                        for end in spaces + [len(line)]:
                            row.append(line[start:end].strip())
                            start = end
                        rows.append(row)
                    return rows[0], rows[1:]
            
            return [], []

        def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
            """Create properly formatted markdown table"""
            if not headers or not rows:
                return response
            
            # Clean and standardize cell content
            headers = [str(h).strip() for h in headers]
            rows = [[str(cell).strip() for cell in row] for row in rows]
            
            # Calculate column widths
            widths = []
            for i in range(len(headers)):
                column = [headers[i]] + [row[i] for row in rows if i < len(row)]
                widths.append(max(len(cell) for cell in column))
            
            # Format table
            header_line = '| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, widths)) + ' |'
            separator = '|' + '|'.join('-' * (w + 2) for w in widths) + '|'
            data_lines = [
                '| ' + ' | '.join(cell.ljust(w) for cell, w in zip(row, widths)) + ' |'
                for row in rows
            ]
            
            return '\n'.join([header_line, separator] + data_lines)

        try:
            headers, rows = parse_table_data(response)
            if headers and rows:
                return format_markdown_table(headers, rows)
            return response
        except Exception as e:
            self.logger.error(f"Error formatting table: {str(e)}")
            return response

    def _format_chart_minimal(self, response: str) -> str:
        """Format chart data with ASCII/Unicode visualization"""
        def extract_chart_data(text: str) -> Dict[str, List[Tuple[str, float]]]:
            data = {
                "time_series": [],
                "percentages": [],
                "quantities": [],
                "comparisons": []
            }
            
            # Extract time series data
            time_series = re.finditer(self.format_patterns["chart"][0], text)
            for match in time_series:
                date = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", match.group())
                value = re.search(r"(?:\d+(?:\.\d+)?)", match.group())
                if date and value:
                    data["time_series"].append((date.group(), float(value.group())))
            
            # Extract percentages
            percentages = re.finditer(self.format_patterns["chart"][1], text)
            for match in percentages:
                value = float(match.group().rstrip('%'))
                data["percentages"].append((match.group(), value))
            
            # Extract quantities
            quantities = re.finditer(self.format_patterns["chart"][2], text)
            for match in quantities:
                value = float(re.search(r"\d+(?:\.\d+)?", match.group()).group())
                data["quantities"].append((match.group(), value))
            
            return data

        def create_ascii_chart(data: Dict[str, List[Tuple[str, float]]]) -> str:
            if not any(data.values()):
                return response
            
            chart_lines = ["```chart"]
            
            # Handle time series
            if data["time_series"]:
                dates, values = zip(*sorted(data["time_series"]))
                max_val = max(values)
                scale = 40 / max_val
                
                chart_lines.append("Time Series Data:")
                for date, val in zip(dates, values):
                    bar = '█' * int(val * scale)
                    chart_lines.append(f"{date} {val:>8.1f} | {bar}")
            
            # Handle percentages
            if data["percentages"]:
                chart_lines.append("\nPercentage Distribution:")
                for label, val in data["percentages"]:
                    bar = '█' * int(val * 0.4)  # Scale to 40 chars max
                    chart_lines.append(f"{label:>8} | {bar}")
            
            # Handle quantities
            if data["quantities"]:
                chart_lines.append("\nQuantity Comparison:")
                max_val = max(val for _, val in data["quantities"])
                scale = 40 / max_val
                for label, val in data["quantities"]:
                    bar = '█' * int(val * scale)
                    chart_lines.append(f"{label:>8} | {bar}")
            
            chart_lines.append("```")
            return '\n'.join(chart_lines)

        try:
            chart_data = extract_chart_data(response)
            if any(chart_data.values()):
                return create_ascii_chart(chart_data)
            return response
        except Exception as e:
            self.logger.error(f"Error formatting chart: {str(e)}")
            return response

    def _format_markdown_minimal(self, response: str) -> str:
        """Apply minimal markdown formatting while preserving content"""
        lines = response.strip().split('\n')
        formatted_lines = []
        in_list = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                formatted_lines.append('')
                in_list = False
                continue
                
            # Preserve existing markdown
            if any(stripped.startswith(md) for md in ['#', '-', '*', '>', '```', '|']):
                formatted_lines.append(line)
                in_list = stripped.startswith(('-', '*'))
                continue
            
            # Format headings (only if it looks like a title)
            if i == 0 or (i > 0 and not lines[i-1].strip()):
                if len(stripped) < 50 and stripped[0].isupper():
                    if i == 0:  # Main title
                        formatted_lines.append(f"# {stripped}")
                    else:  # Subtitle
                        formatted_lines.append(f"## {stripped}")
                    continue
            
            # Format lists (detect potential list items)
            if not in_list and i > 0 and lines[i-1].strip() and (
                stripped.startswith(('The ', 'This ', 'A ', 'An ')) or
                any(stripped.lower().startswith(w) for w in ['use ', 'when ', 'how ', 'why '])
            ):
                formatted_lines.append(f"- {stripped}")
                in_list = True
                continue
            
            # Preserve paragraph structure
            formatted_lines.append(stripped)
            in_list = False
        
        return '\n'.join(formatted_lines)
