"""
Main Streamlit application for YouTube Video Insights Platform.
This file serves as the entry point and orchestrates the modular components.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path

# Import configuration and utilities
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import YouTubeInsightsException, QueryError

# Import core components
from src.video.video_processor import VideoProcessor
from src.query.query_engine import QueryEngine
from src.client.client_manager import ClientManager
from src.dashboard.client_dashboard import ClientDashboard

# Import UI modules
from src.ui.ui_components import (
    load_custom_css, render_header, render_sidebar, render_video_input_section,
    render_video_processing, render_query_result, render_footer
)
from src.ui.authentication import render_authentication_page, check_session_timeout
from src.ui.dashboard import (
    render_client_dashboard_tab, render_settings_tab, render_admin_panel
)

# Set page configuration
st.set_page_config(
    page_title=settings.app_name,
    page_icon=settings.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': f'{settings.app_name} v{settings.app_version}'
    }
)


def initialize_session_state():
    """Initialize all session state variables."""
    # Core application state
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = QueryEngine()
    
    if 'client_manager' not in st.session_state:
        st.session_state.client_manager = ClientManager()
    
    if 'client_dashboard' not in st.session_state:
        st.session_state.client_dashboard = ClientDashboard(st.session_state.client_manager)
    
    # Video processing state
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    
    if 'current_video_url' not in st.session_state:
        st.session_state.current_video_url = ""
    
    # Query state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'last_query_result' not in st.session_state:
        st.session_state.last_query_result = None
    
    # Authentication state (initialized in authentication module)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'client' not in st.session_state:
        st.session_state.client = None
    
    if 'client_session_id' not in st.session_state:
        st.session_state.client_session_id = None
    
    # Performance tracking
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = []
    
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.utcnow()
    
    # User permissions
    if 'user_permissions' not in st.session_state:
        st.session_state.user_permissions = ['basic']
    
    if 'session_timeout' not in st.session_state:
        st.session_state.session_timeout = 24 * 60 * 60  # 24 hours


def render_query_section():
    """Render the query input and processing section."""
    if not st.session_state.processed_video:
        st.info("Please process a video first to enable queries.")
        return
    
    st.markdown("### Ask Questions About the Video")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of this video?",
        help="Ask any question about the video content"
    )
    
    # Query options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        query_button = st.button("Ask Question", type="primary")
    
    with col2:
        language = st.selectbox("Response Language", ["auto", "en", "es", "fr", "de"], index=0)
    
    with col3:
        detailed = st.checkbox("Detailed Response", value=False)
    
    # Process query
    if query_button and query.strip():
        process_query(query, language, detailed)


def process_query(query: str, language: str = "auto", detailed: bool = False):
    """Process a user query and display results."""
    try:
        start_time = datetime.utcnow()
        
        with st.spinner("Processing your question..."):
            # Get the query result
            result = st.session_state.query_engine.query(
                query=query,
                processed_video=st.session_state.processed_video,
                language=language,
                detailed_response=detailed
            )
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Add to query history
            st.session_state.query_history.append((query, datetime.utcnow()))
            st.session_state.last_query_result = result
            
            # Log analytics (if authenticated)
            if st.session_state.authenticated and st.session_state.client:
                try:
                    st.session_state.client_manager.log_query_analytics(
                        client_id=st.session_state.client.client_id,
                        session_id=st.session_state.client_session_id,
                        query=query,
                        response_time=response_time,
                        confidence_score=result.confidence_score,
                        video_id=st.session_state.processed_video.video_info.video_id
                    )
                    
                    # Update API usage
                    st.session_state.client_manager.update_api_usage(st.session_state.client.client_id)
                    
                except Exception as e:
                    logger.error(f"Failed to log analytics: {str(e)}")
            
            # Display result
            render_query_result(result, response_time)
            
    except QueryError as e:
        st.error(f"ERROR: Query processing error: {str(e)}")
        logger.error(f"Query error: {str(e)}")
    except Exception as e:
        st.error(f"ERROR: Unexpected error: {str(e)}")
        logger.error(f"Unexpected query error: {str(e)}")


def render_video_analysis_tab():
    """Render the main video analysis interface."""
    # Load custom CSS
    load_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Video input section
    video_url, process_button, use_cache = render_video_input_section()
    
    # Process video if requested
    if process_button and video_url:
        render_video_processing(video_url, use_cache)
    
    # Query section
    render_query_section()
    
    # Footer
    render_footer()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_main_interface():
    """Render the main application interface with tabs."""
    # Check if user has admin permissions
    has_admin = 'admin' in st.session_state.user_permissions
    
    # Create navigation tabs
    if has_admin:
        tab1, tab2, tab3, tab4 = st.tabs(["Video Analysis", "Dashboard", "Settings", "Admin"])
        
        with tab4:
            render_admin_panel()
    else:
        tab1, tab2, tab3 = st.tabs(["Video Analysis", "Dashboard", "Settings"])
    
    with tab1:
        render_video_analysis_tab()
    
    with tab2:
        render_client_dashboard_tab()
    
    with tab3:
        render_settings_tab()


def validate_client_session():
    """Validate the current client session."""
    if st.session_state.client_session_id:
        client = st.session_state.client_manager.validate_session(st.session_state.client_session_id)
        if not client:
            st.session_state.authenticated = False
            st.session_state.client = None
            st.session_state.client_session_id = None
            st.rerun()
        else:
            st.session_state.client = client


def main():
    """Main application entry point."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Load custom CSS
        load_custom_css()
        
        # Check for session timeout
        if check_session_timeout():
            return
        
        # Check authentication
        if not st.session_state.authenticated:
            if not render_authentication_page():
                return
        
        # Validate client session
        validate_client_session()
        
        # Render main interface
        render_main_interface()
        
    except Exception as e:
        st.error(f"ERROR: Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        if settings.debug:
            st.exception(e)


if __name__ == "__main__":
    main()