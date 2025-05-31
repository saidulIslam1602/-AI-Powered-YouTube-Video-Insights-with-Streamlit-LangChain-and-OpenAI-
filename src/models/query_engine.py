"""Query processing and response generation engine."""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langdetect import detect

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import QueryError, LanguageDetectionError
from src.models.video_processor import ProcessedVideo

@dataclass
class QueryResult:
    """Query result container."""
    query: str
    response: str
    confidence_score: float
    source_chunks: List[str]
    language: str
    processing_time: float
    metadata: Dict[str, Any]

class QueryEngine:
    """Handles query processing and response generation."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
    def detect_query_language(self, query: str) -> str:
        """Detect the language of the user query."""
        try:
            if not query.strip():
                raise LanguageDetectionError("Query is empty")
            
            detected_lang = detect(query)
            logger.info(f"Detected query language: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Query language detection failed: {str(e)}, defaulting to English")
            return 'en'  # Default to English
    
    def search_relevant_chunks(self, processed_video: ProcessedVideo, query: str, k: int = None) -> List[Document]:
        """Search for relevant chunks in the video transcript."""
        try:
            if k is None:
                k = settings.similarity_k
            
            relevant_docs = processed_video.vector_store.similarity_search(query, k=k)
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query: {query}")
                return []
            
            logger.info(f"Found {len(relevant_docs)} relevant document chunks")
            return relevant_docs
            
        except Exception as e:
            raise QueryError(f"Error searching relevant chunks: {str(e)}")
    
    def calculate_confidence_score(self, query: str, chunks: List[str], response: str) -> float:
        """Calculate confidence score for the response."""
        try:
            # Simple heuristic-based confidence calculation
            query_terms = set(query.lower().split())
            
            # Check overlap between query terms and chunks
            chunk_text = " ".join(chunks).lower()
            chunk_overlap = len(query_terms.intersection(set(chunk_text.split()))) / len(query_terms) if query_terms else 0
            
            # Check response quality indicators
            response_length_score = min(len(response) / 500, 1.0)  # Normalize to 500 chars
            
            # Combine scores
            confidence = (chunk_overlap * 0.6) + (response_length_score * 0.4)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {str(e)}")
            return 0.5  # Default confidence
    
    def create_system_prompt(self, language: str, video_language: str = None) -> str:
        """Create system prompt based on detected languages."""
        base_prompt = (
            "You are an intelligent assistant specialized in analyzing YouTube video content. "
            "Your task is to provide accurate, comprehensive answers based on the video transcript provided. "
        )
        
        language_instruction = f"Please respond in {self._get_language_name(language)}. "
        
        guidelines = (
            "\nGuidelines:\n"
            "1. Only use information explicitly stated in the transcript\n"
            "2. If the transcript doesn't contain enough information, clearly state that\n"
            "3. Provide detailed, well-structured responses when possible\n"
            "4. Include relevant timestamps or context when helpful\n"
            "5. If asked about opinions, focus on what the video presenter says\n"
            "6. Be honest about limitations in the available information\n"
        )
        
        return base_prompt + language_instruction + guidelines
    
    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to language name."""
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'he': 'Hebrew'
        }
        return language_map.get(lang_code, lang_code.upper())
    
    def generate_response(self, processed_video: ProcessedVideo, query: str) -> QueryResult:
        """Generate a response to the user query."""
        import time
        start_time = time.time()
        
        try:
            # Detect query language
            query_language = self.detect_query_language(query)
            
            # Search for relevant chunks
            relevant_docs = self.search_relevant_chunks(processed_video, query)
            
            if not relevant_docs:
                return QueryResult(
                    query=query,
                    response="I couldn't find relevant information in the video transcript to answer your question.",
                    confidence_score=0.0,
                    source_chunks=[],
                    language=query_language,
                    processing_time=time.time() - start_time,
                    metadata={"error": "No relevant chunks found"}
                )
            
            # Combine relevant chunks
            source_chunks = [doc.page_content for doc in relevant_docs]
            combined_context = "\n\n".join(source_chunks)
            
            # Create messages for the LLM
            system_prompt = self.create_system_prompt(
                query_language, 
                processed_video.video_info.language
            )
            
            human_prompt = (
                f"Based on the following video transcript excerpts, please answer this question: {query}\n\n"
                f"Transcript excerpts:\n{combined_context}\n\n"
                f"Question: {query}"
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Generate response
            logger.info(f"Generating response for query: {query[:50]}...")
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(query, source_chunks, response_text)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated response in {processing_time:.2f}s with confidence {confidence:.2f}")
            
            return QueryResult(
                query=query,
                response=response_text,
                confidence_score=confidence,
                source_chunks=source_chunks,
                language=query_language,
                processing_time=processing_time,
                metadata={
                    "video_id": processed_video.video_info.video_id,
                    "video_language": processed_video.video_info.language,
                    "num_chunks_used": len(source_chunks),
                    "total_context_length": len(combined_context)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise QueryError(f"Failed to generate response: {str(e)}")
    
    def batch_process_queries(self, processed_video: ProcessedVideo, queries: List[str]) -> List[QueryResult]:
        """Process multiple queries for the same video."""
        results = []
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
                result = self.generate_response(processed_video, query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {str(e)}")
                # Add error result
                results.append(QueryResult(
                    query=query,
                    response=f"Error processing query: {str(e)}",
                    confidence_score=0.0,
                    source_chunks=[],
                    language='en',
                    processing_time=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def get_query_suggestions(self, processed_video: ProcessedVideo) -> List[str]:
        """Generate suggested questions based on video content."""
        try:
            # Use first few chunks to generate suggestions
            sample_text = " ".join(processed_video.chunks[:3])
            
            system_prompt = (
                "Based on the following video transcript excerpt, generate 5 thoughtful questions "
                "that a viewer might want to ask about this content. Focus on key topics, "
                "main points, and interesting details mentioned."
            )
            
            human_prompt = f"Video transcript excerpt:\n{sample_text}\n\nGenerate 5 relevant questions:"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            suggestions_text = response.content
            
            # Parse suggestions (assuming they're returned as numbered list)
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean up
                    suggestion = line.split('.', 1)[-1].strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:5]  # Return at most 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {str(e)}")
            return [
                "What are the main topics discussed in this video?",
                "Can you summarize the key points?",
                "What are the most important takeaways?",
                "Are there any specific examples mentioned?",
                "What conclusions does the speaker reach?"
            ] 