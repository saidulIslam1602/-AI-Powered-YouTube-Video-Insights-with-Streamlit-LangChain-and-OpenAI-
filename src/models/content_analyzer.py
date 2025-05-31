"""Advanced content analysis for YouTube videos."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import spacy

from src.utils.logger import logger
from src.models.video_processor import ProcessedVideo

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    emotion_scores: Dict[str, float]
    sentiment_label: str  # positive, negative, neutral

@dataclass
class TopicAnalysis:
    """Topic modeling results."""
    main_topics: List[Dict[str, Any]]
    topic_distribution: List[float]
    key_phrases: List[str]
    topic_coherence: float

@dataclass
class ContentClassification:
    """Content classification results."""
    category: str
    confidence: float
    subcategories: List[Dict[str, float]]
    content_type: str  # educational, entertainment, news, etc.

@dataclass
class ContentInsights:
    """Comprehensive content insights."""
    sentiment: SentimentAnalysis
    topics: TopicAnalysis
    classification: ContentClassification
    readability_score: float
    key_entities: List[Dict[str, Any]]
    summary: str
    word_cloud_data: Dict[str, int]

class ContentAnalyzer:
    """Advanced content analysis for video transcripts."""
    
    def __init__(self):
        self.nlp = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Initialized spaCy NLP model")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Perform sentiment analysis on text."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment_label = "positive"
            elif polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Basic emotion detection (simplified)
            emotion_scores = self._analyze_emotions(text)
            
            return SentimentAnalysis(
                polarity=polarity,
                subjectivity=subjectivity,
                emotion_scores=emotion_scores,
                sentiment_label=sentiment_label
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentAnalysis(
                polarity=0.0,
                subjectivity=0.0,
                emotion_scores={},
                sentiment_label="neutral"
            )
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Basic emotion analysis using keyword matching."""
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'amazing', 'wonderful'],
            'anger': ['angry', 'furious', 'annoyed', 'irritated', 'mad', 'upset'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'melancholy'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords) / len(text.split())
            emotion_scores[emotion] = min(score * 100, 1.0)  # Normalize to 0-1
        
        return emotion_scores
    
    def analyze_topics(self, text: str, num_topics: int = 5) -> TopicAnalysis:
        """Perform topic modeling on text."""
        try:
            # Prepare text for topic modeling
            sentences = text.split('. ')
            if len(sentences) < num_topics:
                num_topics = max(1, len(sentences) // 2)
            
            # TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # LDA topic modeling
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            main_topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = topic[top_words_idx[0]]
                
                main_topics.append({
                    'id': topic_idx,
                    'words': top_words[:5],
                    'weight': float(topic_weight),
                    'coherence': self._calculate_topic_coherence(top_words[:5], text)
                })
            
            # Get topic distribution for the document
            doc_topic_dist = lda.transform(self.vectorizer.transform([text]))[0]
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(text)
            
            # Calculate overall coherence
            topic_coherence = np.mean([topic['coherence'] for topic in main_topics])
            
            return TopicAnalysis(
                main_topics=main_topics,
                topic_distribution=doc_topic_dist.tolist(),
                key_phrases=key_phrases,
                topic_coherence=float(topic_coherence)
            )
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            return TopicAnalysis(
                main_topics=[],
                topic_distribution=[],
                key_phrases=[],
                topic_coherence=0.0
            )
    
    def _calculate_topic_coherence(self, words: List[str], text: str) -> float:
        """Calculate topic coherence (simplified)."""
        text_lower = text.lower()
        word_counts = [text_lower.count(word) for word in words]
        return np.mean(word_counts) / len(text.split()) if word_counts else 0.0
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
            
            # Filter and rank by frequency
            phrase_counts = {}
            for phrase in noun_phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
            # Return top phrases
            sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
            return [phrase for phrase, count in sorted_phrases[:10]]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def classify_content(self, text: str) -> ContentClassification:
        """Classify content type and category."""
        try:
            # Educational keywords
            educational_keywords = [
                'learn', 'tutorial', 'explain', 'guide', 'lesson', 'course',
                'study', 'teach', 'education', 'instruction', 'training'
            ]
            
            # Entertainment keywords
            entertainment_keywords = [
                'funny', 'entertainment', 'fun', 'comedy', 'music', 'game',
                'movie', 'show', 'performance', 'art', 'creative'
            ]
            
            # News keywords
            news_keywords = [
                'news', 'report', 'breaking', 'update', 'announcement',
                'press', 'journalism', 'current', 'events', 'politics'
            ]
            
            # Technology keywords
            tech_keywords = [
                'technology', 'software', 'computer', 'AI', 'programming',
                'tech', 'digital', 'innovation', 'development', 'coding'
            ]
            
            text_lower = text.lower()
            
            # Calculate scores for each category
            categories = {
                'educational': self._calculate_keyword_score(text_lower, educational_keywords),
                'entertainment': self._calculate_keyword_score(text_lower, entertainment_keywords),
                'news': self._calculate_keyword_score(text_lower, news_keywords),
                'technology': self._calculate_keyword_score(text_lower, tech_keywords)
            }
            
            # Determine primary category
            primary_category = max(categories, key=categories.get)
            confidence = categories[primary_category]
            
            # Create subcategories list
            subcategories = [
                {'name': cat, 'score': score}
                for cat, score in sorted(categories.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # Determine content type
            if categories['educational'] > 0.1:
                content_type = 'educational'
            elif categories['entertainment'] > 0.1:
                content_type = 'entertainment'
            elif categories['news'] > 0.1:
                content_type = 'informational'
            else:
                content_type = 'general'
            
            return ContentClassification(
                category=primary_category,
                confidence=confidence,
                subcategories=subcategories,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Error in content classification: {e}")
            return ContentClassification(
                category='general',
                confidence=0.0,
                subcategories=[],
                content_type='general'
            )
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword density score."""
        total_words = len(text.split())
        keyword_count = sum(text.count(keyword) for keyword in keywords)
        return min(keyword_count / total_words, 1.0) if total_words > 0 else 0.0
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease)."""
        try:
            blob = TextBlob(text)
            sentences = len(blob.sentences)
            words = len(blob.words)
            syllables = sum(self._count_syllables(word) for word in blob.words)
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            return max(0, min(100, score))  # Clamp between 0-100
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def generate_word_cloud_data(self, text: str) -> Dict[str, int]:
        """Generate word frequency data for word cloud."""
        try:
            blob = TextBlob(text)
            words = [word.lower() for word in blob.words if len(word) > 3]
            
            # Remove common stop words
            stop_words = set(['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said'])
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequencies
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Return top 50 words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_words[:50])
            
        except Exception as e:
            logger.error(f"Error generating word cloud data: {e}")
            return {}
    
    def analyze_comprehensive(self, processed_video: ProcessedVideo) -> ContentInsights:
        """Perform comprehensive content analysis."""
        text = processed_video.transcript
        
        logger.info(f"Starting comprehensive analysis for video {processed_video.video_info.video_id}")
        
        # Perform all analyses
        sentiment = self.analyze_sentiment(text)
        topics = self.analyze_topics(text)
        classification = self.classify_content(text)
        readability = self.calculate_readability(text)
        entities = self.extract_entities(text)
        word_cloud_data = self.generate_word_cloud_data(text)
        
        # Generate summary
        summary = self._generate_summary(text, sentiment, topics, classification)
        
        return ContentInsights(
            sentiment=sentiment,
            topics=topics,
            classification=classification,
            readability_score=readability,
            key_entities=entities,
            summary=summary,
            word_cloud_data=word_cloud_data
        )
    
    def _generate_summary(self, text: str, sentiment: SentimentAnalysis, 
                         topics: TopicAnalysis, classification: ContentClassification) -> str:
        """Generate a summary of the content analysis."""
        try:
            summary_parts = []
            
            # Content type and category
            summary_parts.append(f"This is a {classification.content_type} video "
                               f"primarily categorized as {classification.category}.")
            
            # Sentiment
            summary_parts.append(f"The overall sentiment is {sentiment.sentiment_label} "
                               f"with a polarity score of {sentiment.polarity:.2f}.")
            
            # Topics
            if topics.main_topics:
                top_topic_words = ", ".join(topics.main_topics[0]['words'][:3])
                summary_parts.append(f"The main topics discussed include: {top_topic_words}.")
            
            # Readability
            summary_parts.append(f"The content has a readability score of {self.calculate_readability(text):.1f}, "
                               "indicating moderate complexity.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Content analysis completed with basic insights available."

# Global content analyzer instance
content_analyzer = ContentAnalyzer() 