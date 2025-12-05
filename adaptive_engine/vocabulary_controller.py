"""
Vocabulary Controller
Adjusts vocabulary complexity based on user level
"""

class VocabularyController:
    """Adjust and analyze text complexity"""

    def adjust_vocabulary(self, text, user_level, is_struggling):
        """Simplify or keep text complex based on user state"""
        if user_level == "beginner" or is_struggling:
            return text.replace("interconnected", "linked").replace("framework", "structure")
        return text

    def analyze_complexity(self, text):
        """Return dummy complexity analysis"""
        words = text.split()
        complexity_score = min(100, max(0, len([w for w in words if len(w) > 7])))
        recommended_level = "advanced" if complexity_score > 10 else "beginner"
        return {"complexity_score": complexity_score, "recommended_level": recommended_level}

    def generate_vocabulary_guidance(self, user_level):
        """Return guidance string for LLM"""
        if user_level == "beginner":
            return "Use simple language, short sentences, and clear examples."
        elif user_level == "intermediate":
            return "Use balanced language with some technical terms."
        return "Use advanced terminology with in-depth explanations."
