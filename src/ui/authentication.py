"""Authentication functionality for the Streamlit application."""

import streamlit as st
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.config.settings import settings
from src.utils.logger import logger
from src.client.client_manager import ClientManager
from src.models.client import Client


def initialize_auth_session_state() -> None:
    """Initialize authentication-related session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'client' not in st.session_state:
        st.session_state.client = None
    
    if 'client_session_id' not in st.session_state:
        st.session_state.client_session_id = None
    
    if 'user_permissions' not in st.session_state:
        st.session_state.user_permissions = ['basic']
    
    if 'session_timeout' not in st.session_state:
        st.session_state.session_timeout = 24 * 60 * 60  # 24 hours in seconds
    
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.utcnow()


def check_session_timeout() -> bool:
    """Check if the current session has timed out."""
    if not st.session_state.authenticated:
        return False
    
    current_time = datetime.utcnow()
    time_since_activity = (current_time - st.session_state.last_activity).total_seconds()
    
    if time_since_activity > st.session_state.session_timeout:
        logout_user()
        st.warning("Session timed out. Please log in again.")
        return True
    
    # Update last activity
    st.session_state.last_activity = current_time
    return False


def render_login_form() -> Optional[Dict[str, str]]:
    """Render the login form and return credentials if submitted."""
    st.markdown("### Client Login")
    
    # Create a form for authentication
    with st.form("login_form"):
        st.markdown("Please enter your client credentials to access the service.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            client_id = st.text_input(
                "Client ID",
                placeholder="Enter your client ID",
                help="Your unique client identifier"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Enter your API key",
                help="Your secure API access key"
            )
        
        # Additional fields for new client registration
        st.markdown("---")
        st.markdown("**New Client Registration (Optional)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input(
                "Company Name",
                placeholder="Your company name",
                help="For new client registration"
            )
        
        with col2:
            contact_name = st.text_input(
                "Contact Name",
                placeholder="Your full name",
                help="Primary contact person"
            )
        
        subscription_tier = st.selectbox(
            "Subscription Tier",
            ["basic", "premium", "enterprise"],
            help="Select your subscription level"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            login_submitted = st.form_submit_button("Login", type="primary")
        
        with col2:
            register_submitted = st.form_submit_button("Register New Client")
        
        with col3:
            demo_submitted = st.form_submit_button("Demo Mode")
    
    # Process form submissions
    if login_submitted and client_id and api_key:
        return {
            'action': 'login',
            'client_id': client_id,
            'api_key': api_key
        }
    
    elif register_submitted and company_name and contact_name:
        return {
            'action': 'register',
            'company_name': company_name,
            'contact_name': contact_name,
            'subscription_tier': subscription_tier,
            'client_id': client_id,
            'api_key': api_key
        }
    
    elif demo_submitted:
        return {
            'action': 'demo'
        }
    
    return None


def authenticate_client(client_id: str, api_key: str, client_manager: ClientManager) -> bool:
    """Authenticate a client with the given credentials."""
    try:
        client = client_manager.authenticate_client(client_id, api_key)
        
        if client:
            st.session_state.authenticated = True
            st.session_state.client = client
            st.session_state.client_session_id = client_manager.create_session(client_id)
            st.session_state.last_activity = datetime.utcnow()
            
            logger.info(f"Client authenticated: {client_id}")
            return True
        else:
            st.error("Invalid client credentials. Please check your Client ID and API Key.")
            return False
    
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        st.error("Authentication failed. Please try again.")
        return False


def register_new_client(client_data: Dict[str, str], client_manager: ClientManager) -> bool:
    """Register a new client with the provided information."""
    try:
        # Generate client ID and API key if not provided
        if not client_data.get('client_id'):
            client_data['client_id'] = generate_client_id(client_data['company_name'])
        
        if not client_data.get('api_key'):
            client_data['api_key'] = generate_api_key()
        
        # Create new client
        new_client = client_manager.create_client(
            client_id=client_data['client_id'],
            api_key=client_data['api_key'],
            company_name=client_data['company_name'],
            contact_name=client_data['contact_name'],
            subscription_tier=client_data['subscription_tier']
        )
        
        if new_client:
            st.success(f"Client registered successfully!")
            st.info(f"**Client ID:** {client_data['client_id']}")
            st.info(f"**API Key:** {client_data['api_key']}")
            st.warning("Please save these credentials securely. You will need them to log in.")
            
            logger.info(f"New client registered: {client_data['client_id']}")
            return True
        else:
            st.error("Failed to register client. Please try again.")
            return False
    
    except Exception as e:
        logger.error(f"Client registration error: {str(e)}")
        st.error("Registration failed. Please check your information and try again.")
        return False


def setup_demo_mode() -> bool:
    """Set up demo mode with limited functionality."""
    try:
        # Create a demo client object
        demo_client = Client(
            client_id="demo_user",
            company_name="Demo Company",
            contact_name="Demo User",
            subscription_tier="basic",
            api_quota=10,
            api_usage=0,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            is_active=True
        )
        
        st.session_state.authenticated = True
        st.session_state.client = demo_client
        st.session_state.client_session_id = "demo_session"
        st.session_state.user_permissions = ['basic', 'demo']
        st.session_state.last_activity = datetime.utcnow()
        
        st.success("Demo mode activated! Limited to 10 video analyses.")
        st.info("Demo mode provides full functionality with usage limitations.")
        
        logger.info("Demo mode activated")
        return True
    
    except Exception as e:
        logger.error(f"Demo mode setup error: {str(e)}")
        st.error("Failed to set up demo mode. Please try again.")
        return False


def logout_user() -> None:
    """Log out the current user and clear session data."""
    try:
        if st.session_state.authenticated and st.session_state.client:
            logger.info(f"User logged out: {st.session_state.client.client_id}")
        
        # Clear authentication-related session state
        st.session_state.authenticated = False
        st.session_state.client = None
        st.session_state.client_session_id = None
        st.session_state.user_permissions = ['basic']
        
        # Clear other session data
        if 'processed_video' in st.session_state:
            st.session_state.processed_video = None
        if 'current_video_url' in st.session_state:
            st.session_state.current_video_url = ""
        if 'query_history' in st.session_state:
            st.session_state.query_history = []
        
        st.success("Logged out successfully.")
        st.rerun()
    
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        st.error("Error during logout. Please refresh the page.")


def render_user_info() -> None:
    """Render current user information in the sidebar."""
    if st.session_state.authenticated and st.session_state.client:
        st.markdown("### Current User")
        st.markdown(f"**Company:** {st.session_state.client.company_name}")
        st.markdown(f"**User:** {st.session_state.client.contact_name}")
        st.markdown(f"**Client ID:** {st.session_state.client.client_id}")
        
        # Logout button
        if st.button("Logout", type="secondary"):
            logout_user()


def check_user_permissions(required_permission: str) -> bool:
    """Check if the current user has the required permission."""
    if not st.session_state.authenticated:
        return False
    
    return required_permission in st.session_state.user_permissions


def generate_client_id(company_name: str) -> str:
    """Generate a unique client ID based on company name."""
    # Create a hash of company name + timestamp
    timestamp = str(int(datetime.utcnow().timestamp()))
    company_clean = ''.join(c.lower() for c in company_name if c.isalnum())
    
    hash_input = f"{company_clean}_{timestamp}"
    hash_object = hashlib.md5(hash_input.encode())
    
    return f"client_{hash_object.hexdigest()[:8]}"


def generate_api_key() -> str:
    """Generate a secure API key."""
    import secrets
    return f"key_{secrets.token_hex(16)}"


def render_authentication_page() -> bool:
    """Render the complete authentication page and handle login process."""
    initialize_auth_session_state()
    
    # Check for session timeout
    if check_session_timeout():
        return False
    
    # If already authenticated, show user info and return True
    if st.session_state.authenticated:
        return True
    
    # Show login form
    st.markdown("## Authentication Required")
    st.markdown("Please authenticate to access the YouTube Video Insights platform.")
    
    credentials = render_login_form()
    
    if credentials:
        if credentials['action'] == 'login':
            return authenticate_client(
                credentials['client_id'], 
                credentials['api_key'], 
                st.session_state.client_manager
            )
        
        elif credentials['action'] == 'register':
            return register_new_client(credentials, st.session_state.client_manager)
        
        elif credentials['action'] == 'demo':
            return setup_demo_mode()
    
    return False