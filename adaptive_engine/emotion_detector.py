"""
Emotion Detection Module
Analyzes user text to detect emotional state and generates appropriate response tone.
"""

import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

# In production:
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch

class EmotionType(Enum):
    """Supported emotion categories."""
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    BORED = "bored"
    CONFIDENT = "confident"

class ResponseTone(Enum):
    """Appropriate response tones based on emotion."""
    SUPPORTIVE = "supportive"      # For confused/frustrated
    ENCOURAGING = "encouraging"    # For anxious/sad
    DETAILED = "detailed"          # For curious/excited
    SIMPLIFIED = "simplified"      # For confused
    ENTHUSIASTIC = "enthusiastic"  # For excited
    CALM = "calm"                  # For anxious
    DIRECT = "direct"              # For confident
    ENGAGING = "engaging"          # For bored

@dataclass
class EmotionResult:
    """Structured emotion detection result."""
    primary_emotion: EmotionType
    confidence: float
    secondary_emotions: List[Tuple[EmotionType, float]]
    detected_keywords: List[str]
    detected_emojis: List[str]
    suggested_tone: ResponseTone
    intensity: float  # 0-1 scale

class EmotionDetector:
    """
    Detects emotions from text using ML models and rule-based patterns.
    """
    
    def __init__(self, use_advanced_model: bool = True):
        """
        Initialize emotion detector.
        
        Args:
            use_advanced_model: Whether to use transformer model vs rule-based
        """
        self.use_advanced_model = use_advanced_model
        
        # Emotion keyword mappings
        self.emotion_keywords = {
            EmotionType.CONFUSED: [
                "confused", "don't understand", "not clear", "what does", "how does",
                "lost", "unclear", "puzzled", "doesn't make sense", "help me"
            ],
            EmotionType.FRUSTRATED: [
                "frustrating", "annoying", "difficult", "hard", "struggling",
                "can't get", "won't work", "problem", "issue", "stuck"
            ],
            EmotionType.CURIOUS: [
                "interesting", "curious", "tell me more", "explain further",
                "why does", "how come", "fascinating", "wonder"
            ],
            EmotionType.EXCITED: [
                "awesome", "amazing", "great", "excited", "love this",
                "cool", "wow", "fantastic", "brilliant"
            ],
            EmotionType.SAD: [
                "sad", "disappointed", "unhappy", "bad", "not good",
                "poor", "low", "depressed", "upset"
            ],
            EmotionType.ANXIOUS: [
                "worried", "anxious", "nervous", "stress", "pressure",
                "concerned", "afraid", "scared", "tense"
            ],
            EmotionType.BORED: [
                "boring", "dull", "tedious", "monotonous", "repetitive",
                "not interesting", "sleepy", "tired of"
            ],
            EmotionType.CONFIDENT: [
                "got it", "understood", "clear", "easy", "simple",
                "figured out", "know this", "confident", "certain"
            ]
        }
        
        # Emoji to emotion mapping
        self.emoji_mapping = {
            "😕": EmotionType.CONFUSED,
            "😟": EmotionType.CONFUSED,
            "🤔": EmotionType.CONFUSED,
            "😠": EmotionType.FRUSTRATED,
            "😤": EmotionType.FRUSTRATED,
            "🙄": EmotionType.FRUSTRATED,
            "😃": EmotionType.EXCITED,
            "😊": EmotionType.HAPPY,
            "🤩": EmotionType.EXCITED,
            "🥳": EmotionType.EXCITED,
            "😢": EmotionType.SAD,
            "😭": EmotionType.SAD,
            "😔": EmotionType.SAD,
            "😨": EmotionType.ANXIOUS,
            "😰": EmotionType.ANXIOUS,
            "😥": EmotionType.ANXIOUS,
            "😴": EmotionType.BORED,
            "💪": EmotionType.CONFIDENT,
            "😎": EmotionType.CONFIDENT,
            "👍": EmotionType.CONFIDENT
        }
        
        # Emotion to response tone mapping
        self.emotion_tone_mapping = {
            EmotionType.CONFUSED: ResponseTone.SUPPORTIVE,
            EmotionType.FRUSTRATED: ResponseTone.SUPPORTIVE,
            EmotionType.CURIOUS: ResponseTone.DETAILED,
            EmotionType.EXCITED: ResponseTone.ENTHUSIASTIC,
            EmotionType.NEUTRAL: ResponseTone.DIRECT,
            EmotionType.HAPPY: ResponseTone.ENCOURAGING,
            EmotionType.SAD: ResponseTone.ENCOURAGING,
            EmotionType.ANXIOUS: ResponseTone.CALM,
            EmotionType.BORED: ResponseTone.ENGAGING,
            EmotionType.CONFIDENT: ResponseTone.DIRECT
        }
        
        # In production, load model here
        # if use_advanced_model:
        #     self.model = pipeline("sentiment-analysis", 
        #                          model="j-hartmann/emotion-english-distilroberta-base")
    
    def detect_emotion(self, text: str, context: Optional[str] = None) -> EmotionResult:
        """
        Detect emotion from text.
        
        Args:
            text: User input text
            context: Optional previous context for better detection
            
        Returns:
            EmotionResult object
        """
        if not text:
            return self._get_neutral_result()
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract components
        keywords = self._extract_keywords(cleaned_text)
        emojis = self._extract_emojis(text)
        question_mark_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Calculate emotion scores
        emotion_scores = self._calculate_emotion_scores(
            cleaned_text, keywords, emojis, question_mark_count, exclamation_count
        )
        
        # Get primary emotion
        primary_emotion, primary_confidence = max(
            emotion_scores.items(), 
            key=lambda x: x[1]
        )
        
        # Get secondary emotions
        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        secondary_emotions = [
            (emotion, score) 
            for emotion, score in sorted_emotions[1:4] 
            if score > 0.1
        ]
        
        # Determine intensity
        intensity = self._calculate_intensity(
            primary_confidence, 
            question_mark_count,
            exclamation_count,
            len(emojis)
        )
        
        # Get suggested tone
        suggested_tone = self.emotion_tone_mapping.get(
            primary_emotion, 
            ResponseTone.DIRECT
        )
        
        # Apply context adjustment if available
        if context:
            emotion_scores = self._adjust_with_context(
                emotion_scores, context
            )
            # Recalculate primary with context
            primary_emotion, primary_confidence = max(
                emotion_scores.items(), 
                key=lambda x: x[1]
            )
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=float(primary_confidence),
            secondary_emotions=secondary_emotions,
            detected_keywords=keywords,
            detected_emojis=emojis,
            suggested_tone=suggested_tone,
            intensity=float(intensity)
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract emotion-related keywords."""
        found_keywords = []
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                # Use word boundaries for exact matching
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text."""
        # Unicode emoji pattern (simplified)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        return emoji_pattern.findall(text)
    
    def _calculate_emotion_scores(self, text: str, keywords: List[str], 
                                 emojis: List[str], question_count: int, 
                                 exclamation_count: int) -> Dict[EmotionType, float]:
        """Calculate scores for each emotion."""
        scores = {emotion: 0.0 for emotion in EmotionType}
        
        # 1. Score from keywords
        for emotion, emotion_keywords in self.emotion_keywords.items():
            for keyword in emotion_keywords:
                if keyword in text:
                    scores[emotion] += 2.0
                elif any(kw in text for kw in keyword.split()):
                    scores[emotion] += 1.0
        
        # 2. Score from emojis
        for emoji in emojis:
            emotion = self.emoji_mapping.get(emoji)
            if emotion:
                scores[emotion] += 3.0
        
        # 3. Score from punctuation
        if question_count > 2:
            scores[EmotionType.CONFUSED] += question_count * 0.5
            scores[EmotionType.CURIOUS] += question_count * 0.3
        
        if exclamation_count > 1:
            scores[EmotionType.EXCITED] += exclamation_count * 0.5
            scores[EmotionType.FRUSTRATED] += exclamation_count * 0.3
        
        # 4. Use advanced model if available
        if self.use_advanced_model:
            model_scores = self._get_model_scores(text)
            for emotion, score in model_scores.items():
                if emotion in scores:
                    scores[emotion] += score * 5.0
        
        # Normalize scores to 0-1
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}
        
        return scores
    
    def _get_model_scores(self, text: str) -> Dict[EmotionType, float]:
        """Get emotion scores from ML model (placeholder)."""
        # Placeholder for actual model inference
        """
        if hasattr(self, 'model'):
            result = self.model(text)
            # Convert model output to our emotion types
            # This would require mapping between model labels and our EmotionType
            pass
        """
        
        # Return neutral for now
        return {EmotionType.NEUTRAL: 0.5}
    
    def _calculate_intensity(self, confidence: float, question_count: int, 
                            exclamation_count: int, emoji_count: int) -> float:
        """Calculate emotion intensity."""
        # Base intensity from confidence
        intensity = confidence
        
        # Adjust based on punctuation
        intensity += min(question_count * 0.05, 0.2)
        intensity += min(exclamation_count * 0.08, 0.3)
        
        # Adjust based on emojis
        intensity += min(emoji_count * 0.1, 0.4)
        
        return min(intensity, 1.0)
    
    def _adjust_with_context(self, current_scores: Dict[EmotionType, float], 
                            context: str) -> Dict[EmotionType, float]:
        """Adjust emotion scores based on conversation context."""
        # Analyze context for emotion consistency
        context_emotion = self.detect_emotion(context)
        
        # If context shows strong emotion, boost related current emotions
        if context_emotion.confidence > 0.7:
            primary_context_emotion = context_emotion.primary_emotion
            current_scores[primary_context_emotion] *= 1.3
        
        return current_scores
    
    def _get_neutral_result(self) -> EmotionResult:
        """Return neutral result for empty input."""
        return EmotionResult(
            primary_emotion=EmotionType.NEUTRAL,
            confidence=0.5,
            secondary_emotions=[],
            detected_keywords=[],
            detected_emojis=[],
            suggested_tone=ResponseTone.DIRECT,
            intensity=0.3
        )
    
    def get_response_template(self, emotion_result: EmotionResult, 
                             topic: Optional[str] = None) -> Dict:
        """
        Generate response template based on detected emotion.
        
        Args:
            emotion_result: Emotion detection result
            topic: Optional topic for context
            
        Returns:
            Dictionary with response parameters
        """
        templates = {
            ResponseTone.SUPPORTIVE: {
                "opening": "I understand this can be challenging. ",
                "style": "patient and step-by-step",
                "pace": "slower",
                "detail_level": "high",
                "encouragement": True
            },
            ResponseTone.ENCOURAGING: {
                "opening": "Great question! ",
                "style": "positive and motivating",
                "pace": "moderate",
                "detail_level": "moderate",
                "encouragement": True
            },
            ResponseTone.DETAILED: {
                "opening": "Let me explain this in detail. ",
                "style": "comprehensive and thorough",
                "pace": "moderate",
                "detail_level": "very_high",
                "encouragement": False
            },
            ResponseTone.SIMPLIFIED: {
                "opening": "Let me break this down simply. ",
                "style": "clear and basic",
                "pace": "slower",
                "detail_level": "low",
                "encouragement": True
            },
            ResponseTone.ENTHUSIASTIC: {
                "opening": "That's an exciting topic! ",
                "style": "energetic and engaging",
                "pace": "faster",
                "detail_level": "moderate",
                "encouragement": True
            },
            ResponseTone.CALM: {
                "opening": "Take your time, let's work through this. ",
                "style": "reassuring and clear",
                "pace": "slow",
                "detail_level": "moderate",
                "encouragement": True
            },
            ResponseTone.DIRECT: {
                "opening": "",
                "style": "straightforward and concise",
                "pace": "normal",
                "detail_level": "appropriate",
                "encouragement": False
            },
            ResponseTone.ENGAGING: {
                "opening": "Let's make this more interesting! ",
                "style": "interactive and dynamic",
                "pace": "variable",
                "detail_level": "moderate",
                "encouragement": True
            }
        }
        
        template = templates.get(
            emotion_result.suggested_tone, 
            templates[ResponseTone.DIRECT]
        )
        
        # Add emotion-specific adjustments
        if emotion_result.primary_emotion == EmotionType.CONFUSED:
            template["use_analogies"] = True
            template["provide_examples"] = True
        elif emotion_result.primary_emotion == EmotionType.CURIOUS:
            template["include_related_topics"] = True
            template["depth"] = "deep"
        
        # Add topic context if available
        if topic:
            template["topic_context"] = topic
        
        return template
    
    def batch_detect(self, texts: List[str]) -> List[EmotionResult]:
        """Detect emotions for multiple texts."""
        return [self.detect_emotion(text) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = EmotionDetector(use_advanced_model=False)
    
    # Test cases
    test_cases = [
        "I'm really confused about this topic 😕 can you explain differently?",
        "This is amazing! I love learning about AI! 🤩",
        "I'm struggling with this problem and it's really frustrating...",
        "Could you tell me more about neural networks?",
        "😢 I don't think I'll ever understand this"
    ]
    
    print("Emotion Detection Test Results:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect_emotion(text)
        
        print(f"\nTest {i}:")
        print(f"  Text: {text[:50]}...")
        print(f"  Primary Emotion: {result.primary_emotion.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Detected Emojis: {result.detected_emojis}")
        print(f"  Suggested Tone: {result.suggested_tone.value}")
        print(f"  Intensity: {result.intensity:.2f}")
        
        # Get response template
        template = detector.get_response_template(result)
        print(f"  Response Style: {template['style']}")