"""
Memory Manager
Handles:
- user state (level, performance)
- question history
- interaction logs
"""

class MemoryManager:

    def __init__(self):
        self.data = {}

    def get_user_state(self, user_id):
        if user_id not in self.data:
            self.data[user_id] = {
                "level": "beginner",
                "history": [],
                "performance_scores": []
            }
        return self.data[user_id]

    def set_user_level(self, user_id, level):
        """Manually set user level (used by selectbox in Streamlit)"""
        state = self.get_user_state(user_id)
        state["level"] = level

    def get_user_level(self, user_id):
        return self.get_user_state(user_id)["level"]

    def update_user_level(self, user_id, performance_score):
        """Auto update level based on performance"""
        state = self.get_user_state(user_id)
        state["performance_scores"].append(performance_score)
        if len(state["performance_scores"]) > 20:
            state["performance_scores"] = state["performance_scores"][-20:]

        avg_score = sum(state["performance_scores"]) / len(state["performance_scores"])

        if avg_score > 0.75 and state["level"] == "beginner":
            state["level"] = "intermediate"
        elif avg_score > 0.85 and state["level"] == "intermediate":
            state["level"] = "advanced"
        elif avg_score < 0.45 and state["level"] == "advanced":
            state["level"] = "intermediate"
        elif avg_score < 0.3 and state["level"] == "intermediate":
            state["level"] = "beginner"

        return state["level"]

    def get_recent_questions(self, user_id):
        state = self.get_user_state(user_id)
        return [h["question"] for h in state["history"][-5:]]

    def store_interaction(self, user_id, question, answer, performance_score, complexity_score):
        state = self.get_user_state(user_id)
        state["history"].append({"question": question, "answer": answer})
        if len(state["history"]) > 50:
            state["history"] = state["history"][-50:]
        state["performance_scores"].append(performance_score)
        if len(state["performance_scores"]) > 20:
            state["performance_scores"] = state["performance_scores"][-20:]
