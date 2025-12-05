class ExplainDifferently:
    def build_prompts(self, text):
        return {
            "simple": f"Explain this in simple beginner-friendly words:\n{text}",
            "example": f"Explain this using a real-world example:\n{text}",
            "technical": f"Explain this in advanced technical style:\n{text}"
        }
