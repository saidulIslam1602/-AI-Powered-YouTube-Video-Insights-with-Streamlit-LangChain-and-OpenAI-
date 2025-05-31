"""Unit tests for video processor module."""

import pytest
from unittest.mock import Mock, patch
from src.models.video_processor import VideoProcessor, VideoInfo
from src.utils.exceptions import InvalidVideoURLError, TranscriptError


class TestVideoProcessor:
    """Test cases for VideoProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def test_extract_video_id_valid_urls(self):
        """Test video ID extraction from valid YouTube URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        
        for url, expected_id in test_cases:
            assert self.processor.extract_video_id(url) == expected_id
    
    def test_extract_video_id_invalid_urls(self):
        """Test video ID extraction from invalid URLs."""
        invalid_urls = [
            "https://www.example.com",
            "not_a_url",
            "https://youtube.com/invalid",
            "",
        ]
        
        for url in invalid_urls:
            with pytest.raises(InvalidVideoURLError):
                self.processor.extract_video_id(url)
    
    def test_validate_video_url_valid(self):
        """Test URL validation for valid YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        
        for url in valid_urls:
            assert self.processor.validate_video_url(url) is True
    
    def test_validate_video_url_invalid(self):
        """Test URL validation for invalid URLs."""
        invalid_urls = [
            "https://www.example.com",
            "not_a_url",
            "ftp://youtube.com/watch?v=test",
            "",
        ]
        
        for url in invalid_urls:
            assert self.processor.validate_video_url(url) is False
    
    def test_get_video_info_valid_url(self):
        """Test getting video info from valid URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_info = self.processor.get_video_info(url)
        
        assert isinstance(video_info, VideoInfo)
        assert video_info.video_id == "dQw4w9WgXcQ"
        assert video_info.url == url
    
    def test_get_video_info_invalid_url(self):
        """Test getting video info from invalid URL."""
        with pytest.raises(InvalidVideoURLError):
            self.processor.get_video_info("https://www.example.com")
    
    @patch('src.models.video_processor.YouTubeTranscriptApi')
    def test_fetch_transcript_success(self, mock_transcript_api):
        """Test successful transcript fetching."""
        # Mock transcript data
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = [
            {'text': 'Hello world', 'start': 0, 'duration': 2},
            {'text': 'This is a test', 'start': 2, 'duration': 3},
        ]
        mock_transcript.language_code = 'en'
        
        mock_transcript_list = Mock()
        mock_transcript_list.find_transcript.return_value = mock_transcript
        mock_transcript_api.list_transcripts.return_value = mock_transcript_list
        
        transcript, language = self.processor.fetch_transcript("test_video_id")
        
        assert transcript == "Hello world This is a test"
        assert language == 'en'
    
    @patch('src.models.video_processor.YouTubeTranscriptApi')
    def test_fetch_transcript_no_transcript_found(self, mock_transcript_api):
        """Test transcript fetching when no transcript is available."""
        from youtube_transcript_api import NoTranscriptFound
        
        mock_transcript_api.list_transcripts.side_effect = NoTranscriptFound(
            "test_video_id", [], "No transcripts found"
        )
        
        with pytest.raises(TranscriptError):
            self.processor.fetch_transcript("test_video_id")
    
    def test_detect_language_valid_text(self):
        """Test language detection with valid text."""
        english_text = "This is an English sentence for testing."
        detected_lang = self.processor.detect_language(english_text)
        assert detected_lang == 'en'
    
    def test_detect_language_empty_text(self):
        """Test language detection with empty text."""
        from src.utils.exceptions import LanguageDetectionError
        
        with pytest.raises(LanguageDetectionError):
            self.processor.detect_language("")
    
    def test_create_chunks_valid_text(self):
        """Test text chunking with valid text."""
        text = "This is a test sentence. " * 100  # Long text
        chunks = self.processor.create_chunks(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 1100 for chunk in chunks)  # chunk_size + some overlap
    
    def test_create_chunks_empty_text(self):
        """Test text chunking with empty text."""
        with pytest.raises(TranscriptError):
            self.processor.create_chunks("")
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        video_id = "test_video_id"
        cache_key = self.processor.get_cache_key(video_id)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
    
    def test_cache_operations(self):
        """Test cache operations."""
        # Test cache stats
        stats = self.processor.get_cache_stats()
        assert 'total_entries' in stats
        assert 'valid_entries' in stats
        assert 'cache_enabled' in stats
        
        # Test cache clearing
        self.processor.clear_cache()
        stats_after_clear = self.processor.get_cache_stats()
        assert stats_after_clear['total_entries'] == 0 