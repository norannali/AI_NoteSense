"""
Prompt Builder
Builds system and user prompts based on level
"""
class PromptBuilder:

    def build_system_prompt(self, level):
        if level == "beginner":
            return ("You are a helpful AI tutor. Explain concepts in very simple, clear language "
                    "with small examples and short sentences.")

        elif level == "intermediate":
            return ("You are an AI tutor. Provide balanced explanations with examples and some "
                    "technical depth when needed.")

        elif level == "advanced":
            return ("You are an expert AI educator. Provide deep technical explanations, structured "
                    "reasoning, and advanced insights.")

        return "You are a helpful AI tutor."

    def build_user_prompt(self, question):
        return f"""
Explain this topic in 3 different ways:

1) Simple explanation for total beginners.
2) Explanation with a practical example.
3) Technical explanation with deeper reasoning.

Topic: {question}
"""
