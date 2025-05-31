"""Database models and operations for YouTube Video Insights."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel
import uuid

from src.config.settings import settings
from src.utils.logger import logger

Base = declarative_base()

class VideoRecord(Base):
    """Database model for processed videos."""
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(50), unique=True, nullable=False, index=True)
    url = Column(String(500), nullable=False)
    title = Column(String(500))
    description = Column(Text)
    duration = Column(Integer)  # in seconds
    language = Column(String(10))
    transcript = Column(Text, nullable=False)
    transcript_hash = Column(String(64), nullable=False)  # For cache invalidation
    chunk_count = Column(Integer, nullable=False)
    processing_time = Column(Float)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)

class QueryRecord(Base):
    """Database model for user queries and responses."""
    __tablename__ = "queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100), index=True)  # For future user management
    session_id = Column(String(100), index=True)
    query_text = Column(Text, nullable=False)
    query_language = Column(String(10))
    response_text = Column(Text, nullable=False)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    chunks_used = Column(Integer)
    source_chunks = Column(JSON)  # Store source chunks as JSON
    metadata = Column(JSON)  # Additional metadata
    rating = Column(Integer)  # User feedback (1-5)
    created_at = Column(DateTime, default=datetime.utcnow)

class UsageAnalytics(Base):
    """Database model for usage analytics."""
    __tablename__ = "usage_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, nullable=False, index=True)
    total_videos_processed = Column(Integer, default=0)
    total_queries = Column(Integer, default=0)
    average_processing_time = Column(Float)
    average_confidence_score = Column(Float)
    popular_video_ids = Column(JSON)  # Top video IDs
    popular_query_types = Column(JSON)  # Query categories
    error_count = Column(Integer, default=0)
    cache_hit_rate = Column(Float)
    unique_users = Column(Integer, default=0)

class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url, echo=settings.debug)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def save_video(self, video_data: Dict[str, Any]) -> str:
        """Save processed video to database."""
        with self.get_session() as session:
            try:
                # Check if video already exists
                existing = session.query(VideoRecord).filter_by(
                    video_id=video_data['video_id']
                ).first()
                
                if existing:
                    # Update existing record
                    existing.access_count += 1
                    existing.last_accessed = datetime.utcnow()
                    session.commit()
                    logger.info(f"Updated existing video record: {video_data['video_id']}")
                    return str(existing.id)
                
                # Create new record
                video_record = VideoRecord(**video_data)
                session.add(video_record)
                session.commit()
                session.refresh(video_record)
                
                logger.info(f"Saved new video record: {video_data['video_id']}")
                return str(video_record.id)
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving video: {e}")
                raise
    
    def save_query(self, query_data: Dict[str, Any]) -> str:
        """Save query and response to database."""
        with self.get_session() as session:
            try:
                query_record = QueryRecord(**query_data)
                session.add(query_record)
                session.commit()
                session.refresh(query_record)
                
                logger.info(f"Saved query record: {query_data['query_text'][:50]}...")
                return str(query_record.id)
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving query: {e}")
                raise
    
    def get_video_analytics(self, video_id: str) -> Dict[str, Any]:
        """Get analytics for a specific video."""
        with self.get_session() as session:
            video = session.query(VideoRecord).filter_by(video_id=video_id).first()
            if not video:
                return {}
            
            queries = session.query(QueryRecord).filter_by(video_id=video_id).all()
            
            return {
                'video_id': video_id,
                'total_queries': len(queries),
                'access_count': video.access_count,
                'average_confidence': sum(q.confidence_score for q in queries if q.confidence_score) / len(queries) if queries else 0,
                'average_processing_time': sum(q.processing_time for q in queries if q.processing_time) / len(queries) if queries else 0,
                'last_accessed': video.last_accessed,
                'created_at': video.created_at
            }
    
    def get_popular_videos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular videos by access count."""
        with self.get_session() as session:
            videos = session.query(VideoRecord).order_by(
                VideoRecord.access_count.desc()
            ).limit(limit).all()
            
            return [
                {
                    'video_id': v.video_id,
                    'title': v.title,
                    'access_count': v.access_count,
                    'created_at': v.created_at,
                    'language': v.language
                }
                for v in videos
            ]
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for the last N days."""
        with self.get_session() as session:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Video statistics
            videos_count = session.query(VideoRecord).filter(
                VideoRecord.created_at >= since_date
            ).count()
            
            # Query statistics
            queries = session.query(QueryRecord).filter(
                QueryRecord.created_at >= since_date
            ).all()
            
            total_queries = len(queries)
            avg_confidence = sum(q.confidence_score for q in queries if q.confidence_score) / total_queries if total_queries else 0
            avg_processing_time = sum(q.processing_time for q in queries if q.processing_time) / total_queries if total_queries else 0
            
            return {
                'period_days': days,
                'total_videos_processed': videos_count,
                'total_queries': total_queries,
                'average_confidence_score': round(avg_confidence, 3),
                'average_processing_time': round(avg_processing_time, 3),
                'queries_per_video': round(total_queries / videos_count, 2) if videos_count else 0
            }
    
    def cleanup_old_records(self, days: int = 90):
        """Clean up old records to manage database size."""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Delete old queries
                deleted_queries = session.query(QueryRecord).filter(
                    QueryRecord.created_at < cutoff_date
                ).delete()
                
                # Delete unused videos (no recent queries)
                unused_videos = session.query(VideoRecord).filter(
                    VideoRecord.last_accessed < cutoff_date,
                    VideoRecord.access_count < 5  # Keep popular videos
                ).delete()
                
                session.commit()
                logger.info(f"Cleaned up {deleted_queries} queries and {unused_videos} videos")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error during cleanup: {e}")
                raise

# Pydantic models for API responses
class VideoAnalytics(BaseModel):
    video_id: str
    total_queries: int
    access_count: int
    average_confidence: float
    average_processing_time: float
    last_accessed: datetime
    created_at: datetime

class UsageStats(BaseModel):
    period_days: int
    total_videos_processed: int
    total_queries: int
    average_confidence_score: float
    average_processing_time: float
    queries_per_video: float

class PopularVideo(BaseModel):
    video_id: str
    title: Optional[str]
    access_count: int
    created_at: datetime
    language: Optional[str]

# Global database manager instance
db_manager = DatabaseManager() if hasattr(settings, 'database_url') else None 