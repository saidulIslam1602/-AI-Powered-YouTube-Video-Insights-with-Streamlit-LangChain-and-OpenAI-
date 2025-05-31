"""Custom exception classes for the application."""

class YouTubeInsightsException(Exception):
    """Base exception class for YouTube Insights application."""
    pass

class TranscriptError(YouTubeInsightsException):
    """Raised when there's an error fetching or processing video transcript."""
    pass

class InvalidVideoURLError(YouTubeInsightsException):
    """Raised when the provided YouTube URL is invalid."""
    pass

class VideoTooLongError(YouTubeInsightsException):
    """Raised when the video is too long for processing."""
    pass

class LanguageDetectionError(YouTubeInsightsException):
    """Raised when language detection fails."""
    pass

class EmbeddingError(YouTubeInsightsException):
    """Raised when there's an error creating embeddings."""
    pass

class QueryError(YouTubeInsightsException):
    """Raised when there's an error processing the user query."""
    pass

class ConfigurationError(YouTubeInsightsException):
    """Raised when there's a configuration error."""
    pass 