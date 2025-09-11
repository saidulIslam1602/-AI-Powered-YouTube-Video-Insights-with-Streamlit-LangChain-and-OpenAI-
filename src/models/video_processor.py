"""Video processing and transcript handling."""

import re
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langdetect import detect, DetectorFactory

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import (
    TranscriptError, InvalidVideoURLError, VideoTooLongError,
    LanguageDetectionError, EmbeddingError
)

# Set seed for consistent language detection
DetectorFactory.seed = 0

@dataclass
class VideoInfo:
    """Video information container."""
    video_id: str
    url: str
    title: Optional[str] = None
    duration: Optional[int] = None  # in seconds
    language: Optional[str] = None

@dataclass
class ProcessedVideo:
    """Processed video data container."""
    video_info: VideoInfo
    transcript: str
    chunks: List[str]
    vector_store: FAISS
    processed_at: datetime

class VideoProcessor:
    """Handles YouTube video processing and transcript analysis."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self._cache: Dict[str, ProcessedVideo] = {}
        
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.info(f"Extracted video ID: {video_id}")
                return video_id
        
        raise InvalidVideoURLError(f"Invalid YouTube URL: {url}")
    
    def validate_video_url(self, url: str) -> bool:
        """Validate YouTube URL format."""
        youtube_patterns = [
            r'https?://(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
            r'https?://youtu\.be/',
            r'https?://m\.youtube\.com/',
            r'https?://music\.youtube\.com/'
        ]
        
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def get_video_info(self, url: str) -> VideoInfo:
        """Get basic video information."""
        if not self.validate_video_url(url):
            raise InvalidVideoURLError(f"Invalid YouTube URL format: {url}")
        
        video_id = self.extract_video_id(url)
        
        return VideoInfo(
            video_id=video_id,
            url=url
        )
    
    def fetch_transcript(self, video_id: str) -> Tuple[str, str]:
        """Fetch video transcript and detect language."""
        try:
            # Create API instance and get available transcripts
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            logger.info(f"Available transcript languages: {[t.language_code for t in transcript_list]}")
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                transcript_entries = transcript.fetch()
                language = transcript.language_code
            except NoTranscriptFound:
                # Get any available transcript
                try:
                    transcript = next(iter(transcript_list))
                    transcript_entries = transcript.fetch()
                    language = transcript.language_code
                except (StopIteration, AttributeError):
                    raise TranscriptError(f"No transcripts available for video {video_id}")
            
            # Handle both old and new transcript API formats
            transcript_text = ""
            estimated_duration = 0
            
            for entry in transcript_entries:
                # Check if entry is a dict or an object with attributes
                if isinstance(entry, dict):
                    transcript_text += entry.get('text', '') + " "
                    estimated_duration = max(estimated_duration, 
                                           entry.get('start', 0) + entry.get('duration', 0))
                else:
                    # Handle object with attributes
                    transcript_text += getattr(entry, 'text', '') + " "
                    start_time = getattr(entry, 'start', 0)
                    duration = getattr(entry, 'duration', 0)
                    estimated_duration = max(estimated_duration, start_time + duration)
            
            transcript_text = transcript_text.strip()
            
            # Validate transcript length
            if len(transcript_text) == 0:
                raise TranscriptError("Transcript is empty")
            
            # Check video duration limits
            max_duration_seconds = settings.max_video_length_minutes * 60
            if estimated_duration > max_duration_seconds:
                raise VideoTooLongError(
                    f"Video is too long ({estimated_duration/60:.1f} minutes). "
                    f"Maximum allowed: {settings.max_video_length_minutes} minutes"
                )
            
            logger.info(f"Successfully fetched transcript in {language} ({len(transcript_text)} characters)")
            return transcript_text, language
            
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            raise TranscriptError(f"No transcript available for video {video_id}: {str(e)}")
        except Exception as e:
            raise TranscriptError(f"Error fetching transcript for video {video_id}: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """Detect language of the given text."""
        try:
            # Use first 1000 characters for detection
            sample_text = text[:1000].strip()
            if not sample_text:
                raise LanguageDetectionError("Text is empty or too short for language detection")
            
            detected_lang = detect(sample_text)
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            raise LanguageDetectionError(f"Failed to detect language: {str(e)}")
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        try:
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                raise ValueError("No chunks created from text")
            
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            raise TranscriptError(f"Error creating text chunks: {str(e)}")
    
    def create_vector_store(self, chunks: List[str]) -> FAISS:
        """Create FAISS vector store from text chunks."""
        try:
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            if not documents:
                raise EmbeddingError("No documents to create embeddings")
            
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            raise EmbeddingError(f"Error creating vector store: {str(e)}")
    
    def get_cache_key(self, video_id: str) -> str:
        """Generate cache key for video."""
        return hashlib.md5(f"{video_id}_{settings.chunk_size}_{settings.chunk_overlap}".encode()).hexdigest()
    
    def is_cache_valid(self, processed_video: ProcessedVideo) -> bool:
        """Check if cached data is still valid."""
        if not settings.cache_enabled:
            return False
        
        age = datetime.now() - processed_video.processed_at
        return age < timedelta(seconds=settings.cache_ttl_seconds)
    
    def process_video(self, url: str, use_cache: bool = True) -> ProcessedVideo:
        """Process a YouTube video completely."""
        video_info = self.get_video_info(url)
        cache_key = self.get_cache_key(video_info.video_id)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_video = self._cache[cache_key]
            if self.is_cache_valid(cached_video):
                logger.info(f"Using cached data for video {video_info.video_id}")
                return cached_video
            else:
                logger.info(f"Cache expired for video {video_info.video_id}")
                del self._cache[cache_key]
        
        logger.info(f"Processing video: {video_info.video_id}")
        
        # Fetch transcript
        transcript, language = self.fetch_transcript(video_info.video_id)
        video_info.language = language
        
        # Create chunks
        chunks = self.create_chunks(transcript)
        
        # Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Create processed video object
        processed_video = ProcessedVideo(
            video_info=video_info,
            transcript=transcript,
            chunks=chunks,
            vector_store=vector_store,
            processed_at=datetime.now()
        )
        
        # Cache the result
        if settings.cache_enabled:
            self._cache[cache_key] = processed_video
            logger.info(f"Cached processed video {video_info.video_id}")
        
        return processed_video
    
    def clear_cache(self) -> None:
        """Clear the video processing cache."""
        self._cache.clear()
        logger.info("Video processing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_entries = sum(1 for pv in self._cache.values() if self.is_cache_valid(pv))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "cache_enabled": settings.cache_enabled,
            "cache_ttl_seconds": settings.cache_ttl_seconds
        } 