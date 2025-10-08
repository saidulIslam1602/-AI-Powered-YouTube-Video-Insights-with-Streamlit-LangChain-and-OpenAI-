"""Reusable UI components for the Streamlit application."""

import streamlit as st
import time
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import YouTubeInsightsException
from src.models.video_processor import ProcessedVideo
from src.models.query_engine import QueryResult


def load_css_file(css_file: str) -> str:
    """Load CSS from external file."""
    css_path = Path(__file__).parent / css_file
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"CSS file not found: {css_path}")
        return ""


def load_custom_css() -> None:
    """Load Microsoft-style CSS for professional UI."""
    css_content = load_css_file('styles.css')
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the main application header with Microsoft style."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{settings.app_name}</h1>
        <p>Enterprise-grade AI video analysis with Microsoft Azure integration</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render the application sidebar."""
    with st.sidebar:
        # Client information (if authenticated)
        if st.session_state.authenticated and st.session_state.client:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Client Information")
            st.markdown(f"**Company:** {st.session_state.client.company_name}")
            st.markdown(f"**User:** {st.session_state.client.contact_name}")
            st.markdown(f"**Tier:** {st.session_state.client.subscription_tier.title()}")
            st.markdown(f"**API Usage:** {st.session_state.client.api_usage}/{st.session_state.client.api_quota}")
            
            # Quota warning
            usage_percentage = (st.session_state.client.api_usage / st.session_state.client.api_quota) * 100
            if usage_percentage > 80:
                st.warning("WARNING: Quota nearly exceeded")
            elif usage_percentage > 100:
                st.error("ERROR: Quota exceeded")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Application Settings")
        
        # Cache management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Cache Management**")
        
        if st.session_state.video_processor:
            cache_stats = st.session_state.video_processor.get_cache_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Entries", cache_stats['total_entries'])
            with col2:
                st.metric("Valid Entries", cache_stats['valid_entries'])
            
            if st.button("Clear Cache"):
                st.session_state.video_processor.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Application info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Application Info**")
        st.info(f"""
        **Version:** {settings.app_version}
        **Model:** {settings.openai_model}
        **Chunk Size:** {settings.chunk_size}
        **Max Video Length:** {settings.max_video_length_minutes} min
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query history
        if st.session_state.query_history:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**Recent Queries**")
            
            for i, (query, timestamp) in enumerate(st.session_state.query_history[-5:]):
                with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.write(f"**Query:** {query}")
                    st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
            
            st.markdown('</div>', unsafe_allow_html=True)


def render_video_input_section() -> Tuple[str, bool, bool]:
    """Render the video URL input section."""
    st.markdown("### Video Analysis")
    
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        value=st.session_state.current_video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the URL of the YouTube video you want to analyze"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        process_button = st.button("Process Video", type="primary")
    
    with col2:
        use_cache = st.checkbox("Use Cache", value=True, help="Use cached results if available")
    
    with col3:
        if st.session_state.processed_video:
            st.success("Ready")
        else:
            st.warning("No video")
    
    return video_url, process_button, use_cache


def render_video_processing(video_url: str, use_cache: bool) -> None:
    """Handle video processing."""
    try:
        with st.spinner("Processing video... This may take a moment."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Validating URL...")
            progress_bar.progress(20)
            
            processed_video = st.session_state.video_processor.process_video(
                video_url, use_cache=use_cache
            )
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            st.session_state.processed_video = processed_video
            st.session_state.current_video_url = video_url
            
            # Display video info
            render_video_info(processed_video)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("Video processed successfully!")
            
    except YouTubeInsightsException as e:
        st.error(f"ERROR: {str(e)}")
        logger.error(f"Video processing error: {str(e)}")
    except Exception as e:
        st.error(f"ERROR: Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in video processing: {str(e)}")


def render_video_info(processed_video: ProcessedVideo) -> None:
    """Render processed video information."""
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Video ID", processed_video.video_info.video_id)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Language", processed_video.video_info.language.upper())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Text Chunks", len(processed_video.chunks))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Characters", f"{len(processed_video.transcript):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_query_result(result: QueryResult, response_time: float = 0) -> None:
    """Render query result with confidence score and performance metrics."""
    st.markdown('<div class="response-container">', unsafe_allow_html=True)
    
    # Header with confidence
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### Answer")
    
    with col2:
        confidence_class = (
            "confidence-high" if result.confidence_score >= 0.7
            else "confidence-medium" if result.confidence_score >= 0.4
            else "confidence-low"
        )
        st.markdown(f"""
        <div class="confidence-badge {confidence_class}">
            Confidence: {result.confidence_score:.0%}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"Time: *{result.processing_time:.1f}s*")
    
    # Answer
    st.markdown("#### Response:")
    st.write(result.response)
    
    # Metadata
    with st.expander("Response Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Query Metadata:**")
            st.write(f"- **Language:** {result.language}")
            st.write(f"- **Chunks Used:** {result.metadata.get('num_chunks_used', 'N/A')}")
            st.write(f"- **Processing Time:** {result.processing_time:.2f}s")
        
        with col2:
            st.markdown("**Video Metadata:**")
            st.write(f"- **Video ID:** {result.metadata.get('video_id', 'N/A')}")
            st.write(f"- **Video Language:** {result.metadata.get('video_language', 'N/A')}")
            st.write(f"- **Context Length:** {result.metadata.get('total_context_length', 'N/A')}")
        
        # Source chunks
        st.markdown("**Source Text Chunks:**")
        for i, chunk in enumerate(result.source_chunks[:3]):  # Show first 3 chunks
            with st.expander(f"Chunk {i+1}"):
                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
    
    # Client feedback section (only for authenticated users)
    if st.session_state.authenticated and st.session_state.client:
        st.markdown("---")
        st.markdown("### Rate This Response")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            feedback_rating = st.slider("How helpful was this response?", 1, 5, 5, 
                                      help="1 = Not helpful, 5 = Very helpful")
            
            if st.button("Submit Feedback", key=f"feedback_{len(st.session_state.query_history)}"):
                try:
                    st.session_state.client_manager.submit_feedback(
                        client_id=st.session_state.client.client_id,
                        session_id=st.session_state.client_session_id,
                        rating=feedback_rating,
                        feedback_text=""
                    )
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    logger.error(f"Failed to submit feedback: {str(e)}")
                    st.error("Failed to submit feedback. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Render application footer."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        f"Powered by {settings.openai_model} | "
        f"Version {settings.app_version} | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )