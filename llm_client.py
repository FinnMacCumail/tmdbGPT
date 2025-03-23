from openai import OpenAI

class OpenAILLMClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content