"""
Ambiguity Detection & Clarification Module
Identifies vague user queries and generates clarification prompts.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

# In production:
# from sentence_transformers import SentenceTransformer
# import spacy

@dataclass
class AmbiguityResult:
    """Structured result from ambiguity analysis."""
    is_ambiguous: bool
    confidence: float
    ambiguity_types: List[str]  # e.g., ["pronoun_reference", "vague_term", "incomplete"]
    ambiguous_phrases: List[str]
    possible_intents: List[Tuple[str, float]]  # (intent, confidence)
    clarification_questions: List[str]
    suggestions: List[str]

class AmbiguityHandler:
    """
    Detects ambiguous queries and generates clarification prompts.
    """
    
    def __init__(self, use_semantic_similarity: bool = True):
        """
        Initialize ambiguity handler.
        
        Args:
            use_semantic_similarity: Whether to use embedding similarity for intent detection
        """
        self.use_semantic_similarity = use_semantic_similarity
        
        # Ambiguous patterns and keywords
        self.vague_pronouns = {
            "this", "that", "it", "they", "them", "those", "these"
        }
        
        self.vague_terms = {
            "thing", "stuff", "something", "everything", "anything",
            "somewhere", "somehow", "somewhat", "certain", "various"
        }
        
        self.ambiguous_verbs = {
            "do", "make", "get", "have", "put", "take", "go"
        }
        
        self.incomplete_indicators = {
            "how to", "what about", "tell me about", "explain",
            "help with", "question about"
        }
        
        # Common intents in educational context
        self.common_intents = {
            "concept_explanation": [
                "explain", "understand", "what is", "meaning of",
                "definition", "concept"
            ],
            "procedure_help": [
                "how to", "steps for", "process of", "method to",
                "solve", "calculate"
            ],
            "example_request": [
                "example", "instance", "case", "demonstration",
                "illustrate", "show me"
            ],
            "comparison": [
                "difference between", "vs", "compared to",
                "similar to", "contrast"
            ],
            "application": [
                "use of", "apply", "practical", "real-world",
                "implementation"
            ],
            "verification": [
                "is this correct", "check", "verify", "right or wrong",
                "am I right"
            ],
            "deep_dive": [
                "details", "in depth", "advanced", "comprehensive",
                "thorough"
            ]
        }
        
        # In production, load models here
        # if use_semantic_similarity:
        #     self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        #     self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_query(self, query: str, context: Optional[str] = None, 
                     topic: Optional[str] = None) -> AmbiguityResult:
        """
        Analyze query for ambiguity and generate clarification.
        
        Args:
            query: User query
            context: Previous conversation context
            topic: Current topic being discussed
            
        Returns:
            AmbiguityResult object
        """
        if not query or len(query.strip()) < 3:
            return AmbiguityResult(
                is_ambiguous=True,
                confidence=1.0,
                ambiguity_types=["incomplete"],
                ambiguous_phrases=[],
                possible_intents=[],
                clarification_questions=["Could you please provide more details?"],
                suggestions=[]
            )
        
        # Clean query
        cleaned_query = self._preprocess_text(query)
        
        # Detect ambiguity types
        ambiguity_types, ambiguous_phrases = self._detect_ambiguity_types(
            cleaned_query, context
        )
        
        # Determine if ambiguous
        is_ambiguous = len(ambiguity_types) > 0
        
        # Calculate confidence
        confidence = self._calculate_ambiguity_confidence(
            ambiguity_types, ambiguous_phrases, cleaned_query
        )
        
        # Detect possible intents
        possible_intents = self._detect_possible_intents(
            cleaned_query, topic
        )
        
        # Generate clarification questions
        clarification_questions = self._generate_clarification_questions(
            cleaned_query, ambiguity_types, possible_intents, topic
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            cleaned_query, possible_intents, topic
        )
        
        return AmbiguityResult(
            is_ambiguous=is_ambiguous,
            confidence=float(confidence),
            ambiguity_types=ambiguity_types,
            ambiguous_phrases=ambiguous_phrases,
            possible_intents=possible_intents,
            clarification_questions=clarification_questions,
            suggestions=suggestions
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_ambiguity_types(self, query: str, 
                               context: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Detect types of ambiguity in query.
        
        Returns:
            Tuple of (ambiguity_types, ambiguous_phrases)
        """
        ambiguity_types = []
        ambiguous_phrases = []
        
        # Check for vague pronouns without clear antecedents
        pronoun_matches = []
        for pronoun in self.vague_pronouns:
            if re.search(r'\b' + pronoun + r'\b', query):
                pronoun_matches.append(pronoun)
        
        if pronoun_matches:
            # Check if context provides clarity
            if not context or not self._pronoun_resolved_in_context(pronoun_matches[0], context):
                ambiguity_types.append("pronoun_reference")
                ambiguous_phrases.extend(pronoun_matches)
        
        # Check for vague terms
        vague_term_matches = []
        for term in self.vague_terms:
            if re.search(r'\b' + term + r'\b', query):
                vague_term_matches.append(term)
        
        if vague_term_matches:
            ambiguity_types.append("vague_term")
            ambiguous_phrases.extend(vague_term_matches)
        
        # Check for ambiguous verbs
        verb_matches = []
        for verb in self.ambiguous_verbs:
            if re.search(r'\b' + verb + r'\b', query):
                verb_matches.append(verb)
        
        if verb_matches and len(query.split()) < 8:  # Short queries with vague verbs
            ambiguity_types.append("vague_action")
            ambiguous_phrases.extend(verb_matches)
        
        # Check for incomplete queries
        for indicator in self.incomplete_indicators:
            if indicator in query and len(query.split()) < 10:
                if not any(word in query for word in ["?", "explain", "about"] + list(self.vague_terms)):
                    ambiguity_types.append("incomplete_query")
                    ambiguous_phrases.append(indicator)
        
        # Check for missing information patterns
        missing_patterns = [
            (r"(what|which|who).*(is|are)", "missing_subject"),
            (r"how to.*", "missing_object"),
            (r"explain.*", "missing_topic")
        ]
        
        for pattern, ambiguity_type in missing_patterns:
            if re.search(pattern, query):
                words = query.split()
                if len(words) < 5:  # Very short query
                    ambiguity_types.append(ambiguity_type)
                    ambiguous_phrases.append(re.search(pattern, query).group(0))
        
        return ambiguity_types, ambiguous_phrases
    
    def _pronoun_resolved_in_context(self, pronoun: str, context: str) -> bool:
        """
        Check if pronoun reference is clear from context.
        
        Args:
            pronoun: The pronoun to check
            context: Previous conversation context
            
        Returns:
            True if pronoun reference is likely clear
        """
        # Simple implementation - check if noun appears in context
        # In production, use coreference resolution
        
        # Common antecedents for pronouns
        pronoun_antecedents = {
            "it": ["concept", "problem", "topic", "idea", "example"],
            "this": ["concept", "problem", "topic", "idea", "example"],
            "that": ["concept", "problem", "topic", "idea", "example"],
            "they": ["methods", "approaches", "solutions", "examples"],
            "them": ["methods", "approaches", "solutions", "examples"]
        }
        
        if pronoun in pronoun_antecedents:
            possible_antecedents = pronoun_antecedents[pronoun]
            for antecedent in possible_antecedents:
                if antecedent in context.lower():
                    return True
        
        return False
    
    def _calculate_ambiguity_confidence(self, ambiguity_types: List[str], 
                                       ambiguous_phrases: List[str], 
                                       query: str) -> float:
        """Calculate confidence score for ambiguity detection."""
        if not ambiguity_types:
            return 0.0
        
        # Base score from number of ambiguity types
        base_score = min(len(ambiguity_types) * 0.3, 0.9)
        
        # Adjust for number of ambiguous phrases
        phrase_score = min(len(ambiguous_phrases) * 0.15, 0.4)
        
        # Adjust for query length (shorter queries are often more ambiguous)
        word_count = len(query.split())
        if word_count < 5:
            length_score = 0.3
        elif word_count < 10:
            length_score = 0.15
        else:
            length_score = 0.0
        
        # Adjust for question marks (questions with ? are often clearer)
        if "?" in query:
            length_score *= 0.5
        
        total_score = base_score + phrase_score + length_score
        
        return min(total_score, 1.0)
    
    def _detect_possible_intents(self, query: str, 
                                topic: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Detect possible user intents.
        
        Returns:
            List of (intent, confidence) tuples
        """
        intent_scores = {}
        
        # Score each intent based on keyword matching
        for intent, keywords in self.common_intents.items():
            score = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query):
                    score += 2
                elif keyword in query:
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        # If topic is provided, boost topic-related intents
        if topic:
            topic_words = topic.lower().split()
            for intent in intent_scores:
                # Some intents are more likely for certain topics
                if intent in ["deep_dive", "application"]:
                    intent_scores[intent] *= 1.5
        
        # Normalize scores
        if intent_scores:
            max_score = max(intent_scores.values())
            intent_scores = {
                intent: score / max_score
                for intent, score in intent_scores.items()
            }
            
            # Convert to sorted list
            sorted_intents = sorted(
                intent_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep only top 3 intents with confidence > 0.2
            result = [
                (intent, float(confidence))
                for intent, confidence in sorted_intents[:3]
                if confidence > 0.2
            ]
        else:
            result = [("unknown", 0.5)]
        
        return result
    
    def _generate_clarification_questions(self, query: str, 
                                         ambiguity_types: List[str],
                                         possible_intents: List[Tuple[str, float]],
                                         topic: Optional[str] = None) -> List[str]:
        """Generate clarification questions based on ambiguity."""
        questions = []
        
        # Generate questions based on ambiguity types
        for amb_type in ambiguity_types:
            if amb_type == "pronoun_reference":
                questions.extend([
                    f"What specifically does '{self._extract_pronouns(query)[0]}' refer to?",
                    "Could you clarify what you mean by that?"
                ])
            elif amb_type == "vague_term":
                vague_term = self._extract_vague_terms(query)[0]
                questions.append(
                    f"Could you specify what you mean by '{vague_term}'?"
                )
            elif amb_type == "incomplete_query":
                questions.extend([
                    "Could you complete your question?",
                    "What specific aspect would you like me to explain?"
                ])
            elif amb_type == "missing_topic":
                questions.append(
                    "What specific topic or concept would you like me to explain?"
                )
        
        # Generate questions based on possible intents
        if possible_intents and possible_intents[0][0] != "unknown":
            primary_intent = possible_intents[0][0]
            
            if primary_intent == "concept_explanation":
                questions.append(
                    "Which specific concept would you like me to explain?"
                )
            elif primary_intent == "procedure_help":
                questions.append(
                    "What specific procedure or steps are you asking about?"
                )
            elif primary_intent == "example_request":
                questions.append(
                    "What type of example would be most helpful?"
                )
            elif primary_intent == "comparison":
                questions.append(
                    "What two things would you like me to compare?"
                )
        
        # Add topic-specific question if topic is known
        if topic and len(questions) < 2:
            questions.append(
                f"Are you asking about a specific aspect of {topic}?"
            )
        
        # Add general clarification questions if needed
        if not questions:
            questions = [
                "Could you please rephrase or provide more details?",
                "What specific information are you looking for?"
            ]
        
        # Remove duplicates and limit
        unique_questions = []
        seen = set()
        for q in questions:
            if q not in seen:
                unique_questions.append(q)
                seen.add(q)
        
        return unique_questions[:3]  # Return top 3
    
    def _generate_suggestions(self, query: str, 
                             possible_intents: List[Tuple[str, float]],
                             topic: Optional[str] = None) -> List[str]:
        """Generate suggestion prompts for rephrasing."""
        suggestions = []
        
        # Get primary intent
        primary_intent = possible_intents[0][0] if possible_intents else "unknown"
        
        # Generate suggestions based on intent
        suggestion_templates = {
            "concept_explanation": [
                "Try asking: 'Explain [specific concept] in simple terms'",
                "Try: 'What is the definition of [specific term]?'"
            ],
            "procedure_help": [
                "Try: 'What are the steps to [specific action]?'",
                "Try: 'How do I [specific task] step by step?'"
            ],
            "example_request": [
                "Try: 'Can you give me an example of [specific concept]?'",
                "Try: 'Show me a practical application of [topic]'"
            ],
            "comparison": [
                "Try: 'What's the difference between [A] and [B]?'",
                "Try: 'Compare [concept1] with [concept2]'"
            ],
            "unknown": [
                "Try to be more specific about what you need help with",
                "Consider mentioning: what subject, what concept, what problem"
            ]
        }
        
        suggestions.extend(suggestion_templates.get(
            primary_intent, 
            suggestion_templates["unknown"]
        ))
        
        # Add topic-specific suggestions
        if topic:
            suggestions.append(
                f"You might ask about: specific {topic} concepts, examples, or applications"
            )
        
        return suggestions
    
    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract vague pronouns from text."""
        found = []
        for pronoun in self.vague_pronouns:
            if re.search(r'\b' + pronoun + r'\b', text):
                found.append(pronoun)
        return found
    
    def _extract_vague_terms(self, text: str) -> List[str]:
        """Extract vague terms from text."""
        found = []
        for term in self.vague_terms:
            if re.search(r'\b' + term + r'\b', text):
                found.append(term)
        return found
    
    def generate_clarified_response(self, original_query: str, 
                                   user_clarification: str,
                                   topic: Optional[str] = None) -> Dict:
        """
        Generate a response that incorporates user clarification.
        
        Args:
            original_query: Original ambiguous query
            user_clarification: User's clarification response
            topic: Current topic
            
        Returns:
            Dictionary with response components
        """
        # Combine original query with clarification
        combined_text = f"{original_query} {user_clarification}".strip()
        
        # Re-analyze for ambiguity
        new_analysis = self.analyze_query(combined_text, topic=topic)
        
        # Check if ambiguity is resolved
        if not new_analysis.is_ambiguous or new_analysis.confidence < 0.3:
            resolution_status = "resolved"
        else:
            resolution_status = "partially_resolved"
        
        # Generate response acknowledging clarification
        response_parts = []
        
        if resolution_status == "resolved":
            response_parts.append("Thank you for clarifying!")
            response_parts.append(f"Now I understand you're asking about: {user_clarification}")
        else:
            response_parts.append("Thanks for providing more details.")
            if new_analysis.clarification_questions:
                response_parts.append(
                    f"To help me better: {new_analysis.clarification_questions[0]}"
                )
        
        # Add topic context if available
        if topic:
            response_parts.append(f"I'll focus on {topic} related to your question.")
        
        return {
            "status": resolution_status,
            "combined_query": combined_text,
            "response_parts": response_parts,
            "remaining_ambiguity": new_analysis.is_ambiguous,
            "remaining_confidence": new_analysis.confidence
        }
    
    def batch_analyze(self, queries: List[str]) -> List[AmbiguityResult]:
        """Analyze multiple queries for ambiguity."""
        return [self.analyze_query(query) for query in queries]

# Example usage
if __name__ == "__main__":
    # Initialize handler
    handler = AmbiguityHandler(use_semantic_similarity=False)
    
    # Test cases
    test_queries = [
        "Explain this",  # Very ambiguous
        "How do I do it?",  # Vague pronoun
        "Tell me about machine learning",  # Somewhat clear
        "What's the thing about algorithms?",  # Vague term
        "Compare them",  # Very ambiguous
        "Can you help me understand backpropagation in neural networks?"  # Clear
    ]
    
    print("Ambiguity Analysis Test Results:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        result = handler.analyze_query(query, topic="computer science")
        
        print(f"\nQuery {i}: '{query}'")
        print(f"  Ambiguous: {result.is_ambiguous}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Ambiguity Types: {', '.join(result.ambiguity_types) if result.ambiguity_types else 'None'}")
        print(f"  Ambiguous Phrases: {result.ambiguous_phrases}")
        
        if result.possible_intents:
            print(f"  Possible Intents:")
            for intent, confidence in result.possible_intents[:2]:
                print(f"    - {intent}: {confidence:.2f}")
        
        if result.clarification_questions:
            print(f"  Clarification Questions:")
            for q in result.clarification_questions[:2]:
                print(f"    • {q}")
        
        if result.suggestions:
            print(f"  Suggestions:")
            for s in result.suggestions[:2]:
                print(f"    • {s}")
    
    # Test clarification response
    print("\n" + "=" * 60)
    print("Clarification Response Example:")
    print("=" * 60)
    
    original = "Explain this"
    clarification = "I mean explain the concept of gradient descent"
    
    response = handler.generate_clarified_response(original, clarification)
    print(f"\nOriginal: '{original}'")
    print(f"Clarification: '{clarification}'")
    print(f"\nResponse Status: {response['status']}")
    print("Response Parts:")
    for part in response['response_parts']:
        print(f"  • {part}")