"""
Topic Detection Module
Automatically identifies and classifies lecture topics using ML/LLM approaches.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

# In production, these would be actual imports
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch

class TopicCategory(Enum):
    """Predefined topic categories for classification."""
    COMPUTER_SCIENCE = "Computer Science"
    MATHEMATICS = "Mathematics"
    PHYSICS = "Physics"
    BIOLOGY = "Biology"
    CHEMISTRY = "Chemistry"
    ENGINEERING = "Engineering"
    BUSINESS = "Business"
    HUMANITIES = "Humanities"
    OTHER = "Other"

@dataclass
class TopicResult:
    """Structured result from topic detection."""
    primary_topic: TopicCategory
    confidence: float
    keywords: List[str]
    subtopics: List[str]
    alternative_topics: List[Tuple[TopicCategory, float]]

class TopicDetector:
    """
    Detects and classifies topics from lecture content.
    Supports both ML-based and keyword-based approaches.
    """
    
    def __init__(self, use_llm_api: bool = False, model_name: str = "bert-base-uncased"):
        """
        Initialize the topic detector.
        
        Args:
            use_llm_api: Whether to use LLM API (like OpenAI) vs local ML model
            model_name: Pretrained model name for local classification
        """
        self.use_llm_api = use_llm_api
        self.model_name = model_name
        
        # Topic keyword mappings (in production, this would be trained)
        self.topic_keywords = {
            TopicCategory.COMPUTER_SCIENCE: [
                "algorithm", "programming", "code", "software", "database",
                "network", "machine learning", "ai", "python", "java"
            ],
            TopicCategory.MATHEMATICS: [
                "equation", "calculus", "algebra", "geometry", "statistics",
                "probability", "theorem", "proof", "derivative", "integral"
            ],
            TopicCategory.PHYSICS: [
                "force", "energy", "velocity", "quantum", "relativity",
                "thermodynamics", "electromagnetism", "particle", "wave"
            ],
            TopicCategory.BIOLOGY: [
                "cell", "dna", "evolution", "genetics", "organism",
                "ecosystem", "photosynthesis", "respiration", "protein"
            ],
            TopicCategory.ENGINEERING: [
                "design", "system", "mechanical", "electrical", "civil",
                "manufacturing", "structure", "material", "dynamics"
            ]
        }
        
        # In production, initialize model here
        # self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """Load pretrained model for topic classification."""
        # Implementation for production:
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # return model, tokenizer
        pass
    
    def detect_from_text(self, text: str, top_k: int = 3) -> TopicResult:
        """
        Detect topic from lecture text.
        
        Args:
            text: Lecture content
            top_k: Number of top topics to return
            
        Returns:
            TopicResult object with detection results
        """
        if not text or len(text.strip()) < 50:
            return TopicResult(
                primary_topic=TopicCategory.OTHER,
                confidence=0.0,
                keywords=[],
                subtopics=[],
                alternative_topics=[]
            )
        
        # Clean and prepare text
        cleaned_text = self._preprocess_text(text)
        
        if self.use_llm_api:
            return self._detect_with_llm(cleaned_text, top_k)
        else:
            return self._detect_with_keywords(cleaned_text, top_k)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important symbols
        text = re.sub(r'[^\w\s\.\-\+]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_with_keywords(self, text: str, top_k: int) -> TopicResult:
        """Detect topic using keyword matching (fallback method)."""
        topic_scores = {}
        found_keywords = {}
        
        # Score each topic based on keyword matches
        for topic, keywords in self.topic_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Check for keyword presence (exact or partial)
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    score += 2
                    matched_keywords.append(keyword)
                elif keyword in text:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                topic_scores[topic] = score
                found_keywords[topic] = matched_keywords
        
        if not topic_scores:
            return TopicResult(
                primary_topic=TopicCategory.OTHER,
                confidence=0.3,
                keywords=self._extract_keywords(text, 10),
                subtopics=[],
                alternative_topics=[]
            )
        
        # Normalize scores
        max_score = max(topic_scores.values())
        normalized_scores = {
            topic: score / max_score 
            for topic, score in topic_scores.items()
        }
        
        # Sort topics by score
        sorted_topics = sorted(
            normalized_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get primary topic
        primary_topic, primary_confidence = sorted_topics[0]
        
        # Get alternative topics
        alternative_topics = [
            (topic, float(conf)) 
            for topic, conf in sorted_topics[1:top_k]
        ]
        
        # Extract subtopics (key phrases)
        subtopics = self._extract_subtopics(text)
        
        return TopicResult(
            primary_topic=primary_topic,
            confidence=float(primary_confidence),
            keywords=found_keywords.get(primary_topic, []),
            subtopics=subtopics,
            alternative_topics=alternative_topics
        )
    
    def _detect_with_llm(self, text: str, top_k: int) -> TopicResult:
        """
        Detect topic using LLM API (placeholder implementation).
        
        Note: In production, this would call actual LLM API
        """
        # Placeholder for LLM API call
        # Example with OpenAI (commented out):
        """
        import openai
        
        prompt = f"""
        Analyze this lecture content and determine its main topic:
        {text[:2000]}
        
        Return JSON with: primary_topic, confidence (0-1), keywords (list), subtopics (list)
        Topics must be from: {[t.value for t in TopicCategory]}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        """
        
        # For now, use keyword method
        return self._detect_with_keywords(text, top_k)
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract important keywords from text."""
        # Simple implementation - in production use TF-IDF or RAKE
        words = text.split()
        
        # Remove stopwords (basic list)
        stopwords = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'that', 'this', 'it'}
        content_words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(content_words)
        
        return [word for word, _ in word_counts.most_common(n)]
    
    def _extract_subtopics(self, text: str, max_subtopics: int = 5) -> List[str]:
        """Extract potential subtopics from text."""
        # Simple implementation - extract noun phrases
        sentences = text.split('.')
        subtopics = []
        
        for sentence in sentences:
            words = sentence.split()
            if 3 <= len(words) <= 8:  # Reasonable length for subtopic
                # Check if it contains important words
                important_words = {'introduction', 'method', 'result', 'analysis', 
                                  'example', 'theory', 'practice', 'application'}
                if any(word in sentence for word in important_words):
                    subtopics.append(sentence.strip())
                    
            if len(subtopics) >= max_subtopics:
                break
        
        return subtopics
    
    def classify_predefined_topics(self, text: str, candidate_topics: List[str]) -> Dict[str, float]:
        """
        Classify text against predefined candidate topics.
        
        Args:
            text: Text to classify
            candidate_topics: List of possible topics
            
        Returns:
            Dictionary of topic -> confidence score
        """
        scores = {}
        for topic in candidate_topics:
            # Simple matching (can be enhanced with embeddings)
            topic_lower = topic.lower()
            text_lower = text.lower()
            
            # Score based on exact matches and partial matches
            exact_matches = len(re.findall(r'\b' + re.escape(topic_lower) + r'\b', text_lower))
            partial_matches = text_lower.count(topic_lower)
            
            score = exact_matches * 2 + partial_matches * 0.5
            scores[topic] = score / (len(text.split()) / 100)  # Normalize by text length
        
        # Normalize scores to 0-1
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def batch_detect(self, texts: List[str]) -> List[TopicResult]:
        """Detect topics for multiple texts."""
        return [self.detect_from_text(text) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = TopicDetector(use_llm_api=False)
    
    # Example lecture content
    lecture = """
    Machine learning algorithms, particularly neural networks, 
    require extensive training data. Backpropagation is used to 
    optimize weights during training. Deep learning models like 
    CNNs are effective for image recognition tasks.
    """
    
    # Detect topic
    result = detector.detect_from_text(lecture)
    
    print(f"Primary Topic: {result.primary_topic.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Keywords: {result.keywords}")
    print(f"Subtopics: {result.subtopics}")
    
    # Test with predefined topics
    candidates = ["Artificial Intelligence", "Data Science", "Programming"]
    scores = detector.classify_predefined_topics(lecture, candidates)
    print(f"\nCandidate Topic Scores: {scores}")