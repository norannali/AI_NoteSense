import os
import openai
import json

class OpenRouterProvider:
    def __init__(self, model="mistralai/mixtral-8x7b-instruct"):
        self.model = model

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("⚠️ WARNING: OPENROUTER_API_KEY not found!")

        openai.api_key = self.api_key
        openai.base_url = "https://openrouter.ai/api/v1"

    def _safe_extract(self, response):
        """
        Ensure response always returns clean text.
        """
        try:
            if isinstance(response, str):
                return response  # Already safe text

            # Convert to dict if needed
            if hasattr(response, "model_dump"):
                response = response.model_dump()

            content = response["choices"][0]["message"]["content"]
            return content

        except Exception as e:
            return f"[Provider Error] Failed to parse: {str(e)}"

    def _call(self, system_prompt, user_prompt):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return self._safe_extract(response)

        except Exception as e:
            return f"[Provider Error] {str(e)}"

    def generate(self, messages):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return self._safe_extract(response)

        except Exception as e:
            return f"[Provider Error] {str(e)}"

    def summarize(self, text):
        system = "You are a professional summarization engine. Summarize clearly."
        return self._call(system, text)

    def explain(self, text, level="intermediate"):
        system = f"You are an educational tutor. Explain for a {level} student."
        return self._call(system, text)

    def explain_differently(self, text):
        system = """Explain in 3 different ways:
1) Simple
2) With an example
3) Technical"""
        return self._call(system, text)

    def chat(self, text):
        return self._call("You are an adaptive AI tutor.", text)
