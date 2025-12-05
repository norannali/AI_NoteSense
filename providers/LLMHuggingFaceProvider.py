from transformers import pipeline

class LLMProvider:
    def __init__(self, model="google/flan-t5-base"):
        self.model = model
        self.pipeline = pipeline("text2text-generation", model=model)

    def generate(self, system_prompt, user_prompt):
        prompt = f"{system_prompt}\n\n{user_prompt}"
        result = self.pipeline(prompt, max_length=300, do_sample=False)
        return result[0]["generated_text"]

    @property
    def provider(self):
        return "HuggingFace Pipeline"
