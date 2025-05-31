"""Main Streamlit application for YouTube Video Insights."""

import streamlit as st
import time
from typing import Optional
from datetime import datetime

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import (
    YouTubeInsightsException, TranscriptError, InvalidVideoURLError,
    VideoTooLongError, LanguageDetectionError, EmbeddingError, QueryError
)
from src.models.video_processor import VideoProcessor, ProcessedVideo
from src.models.query_engine import QueryEngine, QueryResult

# Page configuration
st.set_page_config(
    page_title=settings.page_title,
    page_icon=settings.page_icon,
    layout=settings.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def load_custom_css():
    """Load custom CSS for enhanced UI."""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --error-color: #d62728;
        --background-color: #f8f9fa;
        --text-color: #212529;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    /* Success/Error styling */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Response styling */
    .response-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = QueryEngine()
    
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_video_url' not in st.session_state:
        st.session_state.current_video_url = ""

def render_header():
    """Render the main application header."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{settings.page_icon} {settings.app_name}</h1>
        <p>AI-Powered YouTube Video Analysis with Advanced Language Processing</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.markdown("### 🛠️ Application Settings")
        
        # Cache management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**📦 Cache Management**")
        
        if st.session_state.video_processor:
            cache_stats = st.session_state.video_processor.get_cache_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Entries", cache_stats['total_entries'])
            with col2:
                st.metric("Valid Entries", cache_stats['valid_entries'])
            
            if st.button("🗑️ Clear Cache"):
                st.session_state.video_processor.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Application info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**📊 Application Info**")
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
            st.markdown("**📝 Recent Queries**")
            
            for i, (query, timestamp) in enumerate(st.session_state.query_history[-5:]):
                with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.write(f"**Query:** {query}")
                    st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_video_input_section():
    """Render the video URL input section."""
    st.markdown("### 🎥 Video Analysis")
    
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        value=st.session_state.current_video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the URL of the YouTube video you want to analyze"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        process_button = st.button("🔄 Process Video", type="primary")
    
    with col2:
        use_cache = st.checkbox("Use Cache", value=True, help="Use cached results if available")
    
    with col3:
        if st.session_state.processed_video:
            st.success("✅ Ready")
        else:
            st.warning("⏳ No video")
    
    return video_url, process_button, use_cache

def render_video_processing(video_url: str, use_cache: bool):
    """Handle video processing."""
    try:
        with st.spinner("🔄 Processing video... This may take a moment."):
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
            
            st.success("✅ Video processed successfully!")
            
    except YouTubeInsightsException as e:
        st.error(f"❌ {str(e)}")
        logger.error(f"Video processing error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in video processing: {str(e)}")

def render_video_info(processed_video: ProcessedVideo):
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

def render_query_section():
    """Render the query input and processing section."""
    if not st.session_state.processed_video:
        st.warning("⚠️ Please process a video first before asking questions.")
        return
    
    st.markdown("### 💬 Ask Questions About the Video")
    
    # Query suggestions
    with st.expander("💡 Suggested Questions"):
        try:
            suggestions = st.session_state.query_engine.get_query_suggestions(
                st.session_state.processed_video
            )
            
            for i, suggestion in enumerate(suggestions):
                if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                    process_query(suggestion)
                    
        except Exception as e:
            st.warning("Could not generate query suggestions.")
            logger.error(f"Error generating suggestions: {str(e)}")
    
    # Manual query input
    query = st.text_area(
        "Your Question:",
        placeholder="Ask anything about the video content...",
        height=100,
        help="Ask detailed questions about the video content. The AI will analyze the transcript to provide accurate answers."
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔍 Get Answer", type="primary", disabled=not query.strip()):
            if query.strip():
                process_query(query)
    
    with col2:
        if st.button("📋 Clear History"):
            st.session_state.query_history.clear()
            st.rerun()

def process_query(query: str):
    """Process a user query and display results."""
    try:
        with st.spinner("🤔 Analyzing your question..."):
            result = st.session_state.query_engine.generate_response(
                st.session_state.processed_video, query
            )
            
            # Add to history
            st.session_state.query_history.append((query, datetime.now()))
            
            # Display result
            render_query_result(result)
            
    except QueryError as e:
        st.error(f"❌ Query processing error: {str(e)}")
        logger.error(f"Query error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected query error: {str(e)}")

def render_query_result(result: QueryResult):
    """Render query result with confidence score and metadata."""
    st.markdown('<div class="response-container">', unsafe_allow_html=True)
    
    # Header with confidence
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 💡 Answer")
    
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
        st.markdown(f"⏱️ *{result.processing_time:.1f}s*")
    
    # Answer
    st.markdown("#### Response:")
    st.write(result.response)
    
    # Metadata
    with st.expander("📊 Response Details"):
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
        st.markdown("**📄 Source Text Chunks:**")
        for i, chunk in enumerate(result.source_chunks[:3]):  # Show first 3 chunks
            with st.expander(f"Chunk {i+1}"):
                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Render header
        render_header()
        
        # Render sidebar
        render_sidebar()
        
        # Main content
        video_url, process_button, use_cache = render_video_input_section()
        
        # Process video if button clicked
        if process_button and video_url:
            render_video_processing(video_url, use_cache)
        
        # Query section
        render_query_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
            f"🚀 Powered by {settings.openai_model} | "
            f"Version {settings.app_version} | "
            "Built with ❤️ using Streamlit"
            "</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        if settings.debug:
            st.exception(e)

if __name__ == "__main__":
    main() 