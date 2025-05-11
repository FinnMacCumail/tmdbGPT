# core/llm/llm_client.py

from openai import OpenAI
import os


class OpenAILLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
