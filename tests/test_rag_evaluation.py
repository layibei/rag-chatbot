import pytest
from datasets import Dataset
from typing import List, Dict
import json
import requests
import time  # 添加 time 模块
from langchain.evaluation import load_evaluator
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import os

# 设置 OpenAI API key 环境变量
# 需要一个有效的 OpenAI API key

# 定义本地服务的基础 URL
BASE_URL = "http://localhost:8080"

def create_evaluation_dataset():
    """Create a sample evaluation dataset"""
    return {
        "question": [
            "1.What are the main components of the Kubernetes control plane?",
            "2.What is the role of the kubelet in a Kubernetes cluster?",
        ],
        "answer": [
            """The main components of the Kubernetes control plane include:
                kube-apiserver: The API server that exposes the Kubernetes API and acts as the front end for the Kubernetes control plane.
                etcd: A consistent and highly available key-value store used as Kubernetes' backing store for all cluster data.
                kube-scheduler: A control plane component that watches for newly created Pods with no assigned node and selects a node for them to run on.
                kube-controller-manager: A control plane component that runs controller processes, such as the Node controller, Job controller, EndpointSlice controller, and ServiceAccount controller.
                cloud-controller-manager: A control plane component that embeds cloud-specific control logic, allowing the cluster to interact with cloud provider APIs.
            """,
            
            """The kubelet is an agent that runs on each node in the cluster. It ensures that containers are running in a Pod by taking a set of PodSpecs and ensuring that the containers described in those PodSpecs are running and healthy. The kubelet does not manage containers that were not created by Kubernetes.""",
        ]
    }

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

def test_rag_evaluation():
    # Create evaluation dataset
    eval_dataset = create_evaluation_dataset()
    
    try:
        # 先检查服务是否运行
        health_check = requests.get(f"{BASE_URL}/docs")
        health_check.raise_for_status()
        
        # Get RAG responses from local service
        responses = get_rag_responses(eval_dataset["question"])
        
        # 使用 OpenAI 评估回答
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("\nEvaluation Results:")
        for i, response in enumerate(responses):
            evaluation_prompt = f"""
            Question: {response['question']}
            Expected Answer: {eval_dataset['answer'][i]}
            Actual Answer: {response['answer']}
            
            Rate the actual answer on relevance (0-10) and explain why:
            """
            
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI evaluation expert. Rate the answer and explain your rating."},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            
            print(f"\nQuestion {i+1}: {response['question']}")
            print(f"Answer: {response['answer']}")
            print(f"Evaluation: {completion.choices[0].message.content}")
            
    except requests.RequestException as e:
        pytest.skip(f"Local service is not running or not accessible: {e}") 