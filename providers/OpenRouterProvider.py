import os  # Standard library to interact with the operating system (e.g., reading environment variables)
import json
from openai import OpenAI

class OpenRouterProvider:
    """
    A provider class to interact with OpenRouter API using the OpenAI SDK.
    It handles initialization, safe response extraction, and specific task generation.
    """

    def __init__(self, model="mistralai/mixtral-8x7b-instruct"):
        """
        Initializes the OpenRouter provider with a specific model and API key.
        """
        self.model = model
        
        # Hardcoded API Key (For testing purposes)
        self.api_key = "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"
        
        # Best Practice: Use 'os.getenv' to securely fetch the key from system environment variables.
        # This prevents saving sensitive keys directly in the code.
        # self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            print("⚠️ WARNING: API Key is missing!")

        # Initialize the OpenAI client with OpenRouter's base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1/"
        )

    def _safe_extract(self, response):
        """
        Helper method to safely extract text content from the API response object.
        """
        try:
            # Case 1: If the response is already a string, return it directly
            if isinstance(response, str):
                return response

            # Case 2: Handle standard OpenAI response objects
            # Extracts content from choices[0].message.content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            
            # Case 3: Fallback conversion to string
            return str(response)

        except Exception as e:
            return f"[Provider Error] Failed to parse: {str(e)}"

    def _call(self, system_prompt, user_prompt):
        """
        Internal method to make a standard chat completion call.
        Takes a system prompt (context) and a user prompt (question).
        """
        try:
            # Use the initialized self.client instead of the direct openai module
            response = self.client.chat.completions.create(
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
        """
        Generates a response based on a custom list of messages.
        Useful for passing chat history or complex conversation structures.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return self._safe_extract(response)

        except Exception as e:
            return f"[Provider Error] {str(e)}"

    # --- Domain Specific Methods ---

    def summarize(self, text):
        """
        Task: Summarizes the provided text clearly.
        """
        system = "You are a professional summarization engine. Summarize clearly."
        return self._call(system, text)

    def explain(self, text, level="intermediate"):
        """
        Task: Explains a concept based on the specified education level.
        Default level is 'intermediate'.
        """
        system = f"You are an educational tutor. Explain for a {level} student."
        return self._call(system, text)

    def explain_differently(self, text):
        """
        Task: Provides three distinct types of explanations (Simple, Example-based, Technical).
        """
        system = """Explain in 3 different ways:
1) Simple
2) With an example
3) Technical"""
        return self._call(system, text)

    def chat(self, text):
        """
        Task: General chat interaction acting as an adaptive AI tutor.
        """
        return self._call("You are an adaptive AI tutor.", text)
    
    def detect_topic(self, text):
        """
        Feature 2: Auto Topic Detection
        Analyzes the text to extract the main academic topic.
        """
        system = "Analyze the text and output ONLY the main academic topic in 3-5 words."
        return self._call(system, text)

    def generate_visual_outline(self, text):
        """
        Feature 6: Visual Outline / Concept Map
        Generates Graphviz DOT code to visualize the concepts.
        """
        system = """
        Create a concept map for this text. 
        Output ONLY valid Graphviz DOT code inside ```dot ... ``` block.
        Do not explain anything. Just code.
        """
        return self._call(system, text)