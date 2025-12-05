import os
import openai

class OpenRouterProvider:
    def __init__(self, model="mistralai/mixtral-8x7b-instruct"):
        self.model = model

        # Load Key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("⚠️ WARNING: OPENROUTER_API_KEY not found!")

        openai.api_key = self.api_key
        openai.base_url = "https://openrouter.ai/api/v1"

    def _call(self, system_prompt, user_prompt):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[Provider Error] {str(e)}"

    def summarize(self, text):
        system = "Summarize this text clearly and concisely."
        return self._call(system, text)

    def explain(self, text, level="intermediate"):
        system = f"Explain the topic for a {level} student in an educational way."
        return self._call(system, text)

    def explain_differently(self, text):
        system = """Explain the topic in 3 different ways:
1) Simple explanation
2) With an example
3) Technical explanation
"""
        return self._call(system, text)

    def chat(self, user_message):
        system = "You are a helpful adaptive tutor."
        return self._call(system, user_message)
