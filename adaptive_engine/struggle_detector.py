"""
Struggle Detector
Analyzes questions to detect struggle patterns
"""

class StruggleDetector:
    """Analyze student struggle based on recent questions"""

    def analyze_question(self, question, recent_questions):
        """Dummy struggle analysis"""
        is_repeated = question in recent_questions
        severity = "high" if is_repeated else "low"
        struggle_points = 10 if is_repeated else 0

        return {
            "is_repeated": is_repeated,
            "severity": severity,
            "struggle_points": struggle_points
        }

    def should_trigger_intervention(self, struggle_points, consecutive_struggles):
        """Decide if intervention is needed"""
        needs_intervention = struggle_points + consecutive_struggles > 15
        actions = ["schedule_review", "provide_extra_examples"] if needs_intervention else []
        return {"needs_intervention": needs_intervention, "actions": actions}

    def detect_question_patterns(self, question_history):
        """Return dummy patterns"""
        return {"repeated_questions": len(question_history)}
