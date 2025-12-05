"""
Response Style Controller
Handles tone, formatting, and emotional adjustments
"""

class ResponseStyleController:
    """Controls response tone and style based on student state"""

    def determine_tone(self, emotion, is_struggling, user_level, is_repeated=False):
    # Example logic
        if is_repeated:
            return "supportive"

        if is_struggling:
            return "supportive"

        if user_level == "beginner":
            return "neutral"

        if user_level == "intermediate":
            return "professional"

        return "expert"


    def format_response(self, content, tone, user_level, is_struggling):
        return f"[{tone} | {user_level}] {content}"

    def generate_system_prompt(self, tone, user_level, is_struggling, repeated_question):
        prompt = f"Tone: {tone}. Level: {user_level}."
        if is_struggling:
            prompt += " Student is struggling, simplify explanation."
        if repeated_question:
            prompt += " Question repeated, be patient."
        return prompt
    
    def adjust_for_emotion(self, content, emotion, user_level):
        if emotion == "confused":
            return f"[Supportive] {content} 🤔"
        elif emotion == "happy":
            return f"[Cheerful] {content} 😄"
        elif emotion == "frustrated":
            return f"[Calm] {content} 😌"
        else:
            return content
