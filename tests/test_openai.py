from openai import OpenAI
import os

def test_openai_simple_completion():
    """Test simple completion with OpenAI"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
    )
    
    print(f"\nQuestion: What is 2+2?")
    print(f"Response: {completion.choices[0].message.content}")