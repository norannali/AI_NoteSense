from .memory_manager import MemoryManager
from .response_style import ResponseStyleController
from .struggle_detector import StruggleDetector
from .vocabulary_controller import VocabularyController
from .prompt_builder import PromptBuilder
from providers.OpenRouterProvider import OpenRouterProvider


class AdaptiveEngine:
    def __init__(self):
        self.memory = MemoryManager()
        self.style = ResponseStyleController()
        self.struggle = StruggleDetector()
        self.vocab = VocabularyController()
        self.prompts = PromptBuilder()
        self.llm = OpenRouterProvider()

    def _build_system_prompt(self, level, tone, struggling):
        struggle_note = "User is struggling. Respond supportively." if struggling else ""
        return f"""
You are an adaptive tutor.
User level: {level}
Tone: {tone}
{struggle_note}
"""

    def process(self, user_id, question, level=None, mode="answer"):

        state = self.memory.get_user_state(user_id)
        user_level = level or state["level"]

        recent_q = self.memory.get_recent_questions(user_id)
        struggle_info = self.struggle.analyze_question(question, recent_q)
        struggling = struggle_info["severity"] == "high"

        tone = self.style.determine_tone(
            emotion="confused" if struggling else "neutral",
            is_struggling=struggling,
            user_level=user_level
        )

        if mode == "summarize":
            output = self.llm.summarize(question)

        elif mode == "explain":
            output = self.llm.explain(question, user_level)

        elif mode == "explain_differently":
            output = self.llm.explain_differently(question)

        else:
            system = self._build_system_prompt(user_level, tone, struggling)
            user_prompt = self.prompts.build_user_prompt(question)

            output = self.llm.generate([
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ])

        if not isinstance(output, str):
            output = str(output)

        final_answer = self.style.format_response(output, tone, user_level, struggling)

        self.memory.store_interaction(
            user_id=user_id,
            question=question,
            answer=final_answer,
            performance_score=50,
            complexity_score=5
        )

        return {
            "response": final_answer,
            "metadata": {
                "level": user_level,
                "tone": tone,
                "struggling": struggling,
                "mode": mode,
                "provider": "OpenRouter"
            }
        }
