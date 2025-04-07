# llm_client.py

from openai import OpenAI
import os

class OpenAILLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": "You are a helpful API assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
