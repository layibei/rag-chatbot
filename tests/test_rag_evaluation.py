import pytest
from typing import List, Dict
import json
import requests
import time
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from config.common_settings import CommonConfig
from langchain_community.chat_models import ChatOllama
from config.llm_config import create_llm
from pathlib import Path

# 设置 OpenAI API key 环境变量
# 需要一个有效的 OpenAI API key

# 定义本地服务的基础 URL
BASE_URL = "http://localhost:8080"

def get_rag_responses(questions: List[str]) -> List[Dict]:
    """Get RAG responses from the local API service"""
    responses = []
    for i, question in enumerate(questions):
        try:
            if i > 0:
                print(f"\nWaiting 30 seconds before next question...")
                time.sleep(30)
                
            print(f"\nProcessing question {i+1}: {question}")
            
            response = requests.post(
                f"{BASE_URL}/chat/completion",
                json={
                    "user_input": question
                },
                headers={
                    "x-user-id": "test-user",
                    "x-session-id": "test-session",
                    "x-request-id": "test-request"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            responses.append({
                "question": question,
                "answer": result["data"].get("answer", ""),
            })
            print(f"Question {i+1} completed")
            
        except requests.RequestException as e:
            print(f"Error calling API: {e}")
            raise
            
    return responses

class SimpleLLMHandler:
    def __init__(self):
        self.config = CommonConfig()
        self.llm = self.config._get_llm_model()  # 直接使用 CommonConfig 中的方法

    def ask(self, question: str) -> str:
        """Simple question-answering without RAG"""
        response = self.llm.invoke(question)
        return response.content

def load_evaluation_dataset():
    """Load evaluation dataset from JSON file"""
    data_file = Path(__file__).parent / "data" / "evaluation_dataset.json"
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_html_report(responses, evaluations):
    """Generate HTML report for evaluation results"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Evaluation Report</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 40px;
            }
            .question { 
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .question h3 {
                color: #2c3e50;
                margin-top: 0;
            }
            .answer {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .evaluation {
                background: #e9ecef;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .timestamp {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>RAG Evaluation Report</h1>
        <div class="timestamp">Generated at: {timestamp}</div>
        {content}
    </body>
    </html>
    """

    question_template = """
    <div class="question">
        <h3>Question {index}: {question}</h3>
        <div class="answer">
            <strong>Expected Answer:</strong><br>
            {expected_answer}
        </div>
        <div class="answer">
            <strong>Actual Answer:</strong><br>
            {actual_answer}
        </div>
        <div class="evaluation">
            <strong>Evaluation:</strong><br>
            {evaluation}
        </div>
    </div>
    """

    content = []
    for i, (response, evaluation) in enumerate(zip(responses, evaluations), 1):
        content.append(question_template.format(
            index=i,
            question=response["question"],
            expected_answer=response["expected_answer"].replace("\n", "<br>"),
            actual_answer=response["answer"].replace("\n", "<br>"),
            evaluation=evaluation.replace("\n", "<br>")
        ))

    from datetime import datetime
    report = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content="\n".join(content)
    )

    # 创建 reports 目录
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # 生成带时间戳的报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"rag_evaluation_report_{timestamp}.html"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report_file

def test_rag_evaluation():
    # Load evaluation dataset from file
    eval_dataset = load_evaluation_dataset()
    
    try:
        # 先检查服务是否运行
        health_check = requests.get(f"{BASE_URL}/docs")
        health_check.raise_for_status()
        
        # Get RAG responses from local service
        responses = get_rag_responses(eval_dataset["question"])
        
        # 初始化简单的 LLM handler
        llm_handler = SimpleLLMHandler()
        
        evaluations = []
        responses_with_expected = []
        
        print("\nEvaluation Results:")
        for i, response in enumerate(responses):
            evaluation_prompt = f"""
            Question: {response['question']}
            Expected Answer: {eval_dataset['answer'][i]}
            Actual Answer: {response['answer']}
            
            Rate the actual answer on relevance (0-10) and explain why:
            """
            
            # 直接使用 LLM 进行评估
            eval_result = llm_handler.ask(evaluation_prompt)
            evaluations.append(eval_result)
            
            # 保存完整的响应信息
            responses_with_expected.append({
                "question": response["question"],
                "answer": response["answer"],
                "expected_answer": eval_dataset["answer"][i]
            })
            
            print(f"\nQuestion {i+1}: {response['question']}")
            print(f"Answer: {response['answer']}")
            print(f"Evaluation: {eval_result}")
        
        # 生成 HTML 报告
        report_file = generate_html_report(responses_with_expected, evaluations)
        print(f"\nHTML report generated: {report_file}")
            
    except requests.RequestException as e:
        pytest.skip(f"Local service is not running or not accessible: {e}") 