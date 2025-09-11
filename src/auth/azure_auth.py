"""Microsoft Azure AD authentication integration."""

import os
import json
import base64
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import AuthenticationError

class AzureADAuth:
    """Microsoft Azure AD authentication handler."""
    
    def __init__(self):
        self.tenant_id = os.getenv('AZURE_TENANT_ID', '')
        self.client_id = os.getenv('AZURE_CLIENT_ID', '')
        self.client_secret = os.getenv('AZURE_CLIENT_SECRET', '')
        self.redirect_uri = os.getenv('AZURE_REDIRECT_URI', 'http://localhost:8501/auth/callback')
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = "https://graph.microsoft.com/.default"
        
    def get_auth_url(self, state: str) -> str:
        """Generate Azure AD authentication URL."""
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': 'openid profile email',
            'state': state,
            'response_mode': 'query'
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.authority}/oauth2/v2.0/authorize?{query_string}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        token_url = f"{self.authority}/oauth2/v2.0/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Token exchange failed: {str(e)}")
            raise AuthenticationError("Failed to exchange code for token")
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Microsoft Graph API."""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {str(e)}")
            raise AuthenticationError("Failed to get user information")
    
    def validate_token(self, token: str) -> bool:
        """Validate JWT token."""
        try:
            # In production, you should validate the token signature
            # For now, we'll just decode it
            decoded = jwt.decode(token, options={"verify_signature": False})
            return decoded.get('exp', 0) > datetime.utcnow().timestamp()
        except jwt.InvalidTokenError:
            return False

class EnhancedAuthManager:
    """Enhanced authentication manager with Microsoft standards."""
    
    def __init__(self):
        self.azure_auth = AzureADAuth()
        self.session_secret = os.getenv('SESSION_SECRET', secrets.token_hex(32))
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def generate_state(self) -> str:
        """Generate secure state parameter for OAuth flow."""
        return secrets.token_urlsafe(32)
    
    def create_session(self, user_info: Dict[str, Any], access_token: str) -> str:
        """Create secure session for authenticated user."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_info.get('id'),
            'email': user_info.get('mail') or user_info.get('userPrincipalName'),
            'name': user_info.get('displayName'),
            'company': user_info.get('companyName', ''),
            'access_token': access_token,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'permissions': self._get_user_permissions(user_info)
        }
        
        self.active_sessions[session_id] = session_data
        logger.info(f"Created session for user: {session_data['email']}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and return session data."""
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        
        # Check if session is expired (24 hours)
        if datetime.utcnow() - session['last_activity'] > timedelta(hours=24):
            del self.active_sessions[session_id]
            return None
            
        # Update last activity
        session['last_activity'] = datetime.utcnow()
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Revoked session: {session_id}")
            return True
        return False
    
    def _get_user_permissions(self, user_info: Dict[str, Any]) -> list:
        """Get user permissions based on Microsoft Graph data."""
        permissions = ['basic']
        
        # Add permissions based on user roles/groups
        if user_info.get('jobTitle', '').lower() in ['manager', 'director', 'vp']:
            permissions.append('admin')
            
        if 'data' in user_info.get('jobTitle', '').lower():
            permissions.append('data_analyst')
            
        return permissions
    
    def get_auth_url(self) -> str:
        """Get Azure AD authentication URL."""
        state = self.generate_state()
        return self.azure_auth.get_auth_url(state)
    
    def handle_callback(self, code: str, state: str) -> Optional[str]:
        """Handle OAuth callback and create session."""
        try:
            # Exchange code for token
            token_data = self.azure_auth.exchange_code_for_token(code)
            access_token = token_data.get('access_token')
            
            if not access_token:
                raise AuthenticationError("No access token received")
            
            # Get user information
            user_info = self.azure_auth.get_user_info(access_token)
            
            # Create session
            session_id = self.create_session(user_info, access_token)
            return session_id
            
        except Exception as e:
            logger.error(f"Authentication callback failed: {str(e)}")
            return None