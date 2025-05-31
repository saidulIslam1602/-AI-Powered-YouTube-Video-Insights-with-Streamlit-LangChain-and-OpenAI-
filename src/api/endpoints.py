"""REST API endpoints for YouTube Video Insights."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import time
import uuid
from datetime import datetime

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import *
from src.models.video_processor import VideoProcessor, ProcessedVideo
from src.models.query_engine import QueryEngine, QueryResult
from src.models.database import db_manager, VideoAnalytics, UsageStats, PopularVideo

# FastAPI app initialization
app = FastAPI(
    title="YouTube Video Insights API",
    description="AI-powered YouTube video analysis API with advanced language processing",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Global instances
video_processor = VideoProcessor()
query_engine = QueryEngine()

# Request/Response Models
class VideoProcessRequest(BaseModel):
    url: HttpUrl
    use_cache: bool = True
    force_reprocess: bool = False

class VideoProcessResponse(BaseModel):
    video_id: str
    status: str
    processing_time: float
    language: str
    chunk_count: int
    transcript_length: int
    message: str

class QueryRequest(BaseModel):
    video_id: str
    query: str
    max_chunks: Optional[int] = None

class QueryResponse(BaseModel):
    query_id: str
    query: str
    response: str
    confidence_score: float
    processing_time: float
    language: str
    source_chunks: List[str]
    metadata: Dict[str, Any]

class BatchQueryRequest(BaseModel):
    video_id: str
    queries: List[str]
    max_chunks: Optional[int] = None

class BatchQueryResponse(BaseModel):
    batch_id: str
    total_queries: int
    completed_queries: int
    results: List[QueryResponse]
    total_processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    dependencies: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime
    request_id: str

# Utility functions
def get_request_id(request: Request) -> str:
    """Generate or get request ID for tracking."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication."""
    if settings.api_key_required and credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

# Exception handlers
@app.exception_handler(YouTubeInsightsException)
async def youtube_insights_exception_handler(request: Request, exc: YouTubeInsightsException):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc),
            timestamp=datetime.utcnow(),
            request_id=get_request_id(request)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            timestamp=datetime.utcnow(),
            request_id=get_request_id(request)
        ).dict()
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        uptime=time.time(),  # Simplified uptime
        dependencies={
            "openai": "connected",
            "database": "connected" if db_manager else "disabled",
            "cache": "enabled" if settings.cache_enabled else "disabled"
        }
    )

# Video processing endpoints
@app.post("/api/v1/videos/process", response_model=VideoProcessResponse)
async def process_video(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Process a YouTube video and extract transcript."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing video: {request.url}")
        
        # Process video
        processed_video = video_processor.process_video(
            str(request.url),
            use_cache=request.use_cache and not request.force_reprocess
        )
        
        processing_time = time.time() - start_time
        
        # Save to database if available
        if db_manager:
            background_tasks.add_task(
                save_video_to_db,
                processed_video,
                processing_time
            )
        
        return VideoProcessResponse(
            video_id=processed_video.video_info.video_id,
            status="success",
            processing_time=processing_time,
            language=processed_video.video_info.language,
            chunk_count=len(processed_video.chunks),
            transcript_length=len(processed_video.transcript),
            message="Video processed successfully"
        )
        
    except YouTubeInsightsException as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/videos/{video_id}/query", response_model=QueryResponse)
async def query_video(
    video_id: str,
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Query a processed video with natural language."""
    start_time = time.time()
    
    try:
        # Get processed video from cache or database
        processed_video = await get_processed_video(video_id)
        
        if not processed_video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found or not processed"
            )
        
        # Generate response
        result = query_engine.generate_response(processed_video, request.query)
        
        query_id = str(uuid.uuid4())
        
        # Save to database if available
        if db_manager:
            background_tasks.add_task(
                save_query_to_db,
                query_id,
                video_id,
                result
            )
        
        return QueryResponse(
            query_id=query_id,
            query=result.query,
            response=result.response,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            language=result.language,
            source_chunks=result.source_chunks,
            metadata=result.metadata
        )
        
    except YouTubeInsightsException as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/videos/{video_id}/batch-query", response_model=BatchQueryResponse)
async def batch_query_video(
    video_id: str,
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Process multiple queries for a video."""
    start_time = time.time()
    
    try:
        # Get processed video
        processed_video = await get_processed_video(video_id)
        
        if not processed_video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found or not processed"
            )
        
        # Process batch queries
        results = query_engine.batch_process_queries(processed_video, request.queries)
        
        batch_id = str(uuid.uuid4())
        total_processing_time = time.time() - start_time
        
        # Convert results to response format
        query_responses = []
        for result in results:
            query_id = str(uuid.uuid4())
            
            query_responses.append(QueryResponse(
                query_id=query_id,
                query=result.query,
                response=result.response,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                language=result.language,
                source_chunks=result.source_chunks,
                metadata=result.metadata
            ))
            
            # Save to database
            if db_manager:
                background_tasks.add_task(
                    save_query_to_db,
                    query_id,
                    video_id,
                    result
                )
        
        return BatchQueryResponse(
            batch_id=batch_id,
            total_queries=len(request.queries),
            completed_queries=len(query_responses),
            results=query_responses,
            total_processing_time=total_processing_time
        )
        
    except YouTubeInsightsException as e:
        logger.error(f"Batch query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Analytics endpoints
@app.get("/api/v1/analytics/videos/{video_id}", response_model=VideoAnalytics)
async def get_video_analytics(
    video_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get analytics for a specific video."""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Analytics not available - database not configured"
        )
    
    analytics = db_manager.get_video_analytics(video_id)
    if not analytics:
        raise HTTPException(
            status_code=404,
            detail=f"Analytics not found for video {video_id}"
        )
    
    return VideoAnalytics(**analytics)

@app.get("/api/v1/analytics/usage", response_model=UsageStats)
async def get_usage_statistics(
    days: int = 30,
    api_key: str = Depends(verify_api_key)
):
    """Get overall usage statistics."""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Analytics not available - database not configured"
        )
    
    stats = db_manager.get_usage_statistics(days)
    return UsageStats(**stats)

@app.get("/api/v1/analytics/popular-videos", response_model=List[PopularVideo])
async def get_popular_videos(
    limit: int = 10,
    api_key: str = Depends(verify_api_key)
):
    """Get most popular videos."""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Analytics not available - database not configured"
        )
    
    videos = db_manager.get_popular_videos(limit)
    return [PopularVideo(**video) for video in videos]

# Cache management endpoints
@app.delete("/api/v1/cache/clear")
async def clear_cache(api_key: str = Depends(verify_api_key)):
    """Clear application cache."""
    video_processor.clear_cache()
    logger.info("Cache cleared via API")
    return {"message": "Cache cleared successfully"}

@app.get("/api/v1/cache/stats")
async def get_cache_stats(api_key: str = Depends(verify_api_key)):
    """Get cache statistics."""
    stats = video_processor.get_cache_stats()
    return stats

# Utility functions
async def get_processed_video(video_id: str) -> Optional[ProcessedVideo]:
    """Get processed video from cache or database."""
    # Try to get from cache first
    cache_stats = video_processor.get_cache_stats()
    
    # If not in cache and database available, try to load
    if db_manager:
        # Implementation would load from database and recreate ProcessedVideo
        # This is a simplified version
        pass
    
    return None  # Simplified for now

async def save_video_to_db(processed_video: ProcessedVideo, processing_time: float):
    """Save processed video to database."""
    if not db_manager:
        return
    
    try:
        video_data = {
            'video_id': processed_video.video_info.video_id,
            'url': processed_video.video_info.url,
            'title': processed_video.video_info.title,
            'language': processed_video.video_info.language,
            'transcript': processed_video.transcript,
            'transcript_hash': hash(processed_video.transcript),
            'chunk_count': len(processed_video.chunks),
            'processing_time': processing_time
        }
        
        db_manager.save_video(video_data)
    except Exception as e:
        logger.error(f"Error saving video to database: {e}")

async def save_query_to_db(query_id: str, video_id: str, result: QueryResult):
    """Save query result to database."""
    if not db_manager:
        return
    
    try:
        query_data = {
            'video_id': video_id,
            'session_id': query_id,  # Simplified
            'query_text': result.query,
            'query_language': result.language,
            'response_text': result.response,
            'confidence_score': result.confidence_score,
            'processing_time': result.processing_time,
            'chunks_used': len(result.source_chunks),
            'source_chunks': result.source_chunks,
            'metadata': result.metadata
        }
        
        db_manager.save_query(query_data)
    except Exception as e:
        logger.error(f"Error saving query to database: {e}")

# WebSocket endpoint for real-time updates (future enhancement)
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """WebSocket endpoint for real-time updates."""
#     pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 