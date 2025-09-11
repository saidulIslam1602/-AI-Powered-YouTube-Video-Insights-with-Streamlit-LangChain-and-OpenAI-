"""SQL Server-based client manager."""

import hashlib
import secrets
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

try:
    from src.database.sql_server import db_manager
    from src.utils.logger import logger
    from src.utils.exceptions import AuthenticationError
except ImportError:
    # Fallback for direct execution
    import logging
    logger = logging.getLogger(__name__)
    from database.sql_server import db_manager
    class AuthenticationError(Exception):
        pass

class SQLClient:
    """Client model for SQL Server."""
    
    def __init__(self, client_id: str, company_name: str, contact_name: str, 
                 contact_email: str, subscription_tier: str, api_quota: int, 
                 api_usage: int, created_at: datetime, is_active: bool = True):
        self.client_id = client_id
        self.company_name = company_name
        self.contact_name = contact_name
        self.contact_email = contact_email
        self.subscription_tier = subscription_tier
        self.api_quota = api_quota
        self.api_usage = api_usage
        self.created_at = created_at
        self.is_active = is_active

class SQLClientManager:
    """SQL Server-based client management system."""
    
    def __init__(self):
        self.db = db_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_client(self, company_name: str, contact_email: str, 
                     contact_name: str, password: str, subscription_tier: str = "basic") -> str:
        """Create a new client."""
        try:
            # Check if email already exists
            if self.get_client_by_email(contact_email):
                raise ValueError("Email already exists")
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Set API quota based on subscription tier
            api_quota = {
                'basic': 1000,
                'professional': 5000,
                'enterprise': 50000
            }.get(subscription_tier, 1000)
            
            # Generate client ID
            client_id = str(uuid.uuid4())
            
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO clients (client_id, company_name, contact_name, contact_email, 
                                       password_hash, subscription_tier, api_quota, api_usage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (client_id, company_name, contact_name, contact_email, 
                      password_hash, subscription_tier, api_quota, 0))
                conn.commit()
            
            logger.info(f"Client created: {contact_email}")
            return client_id
            
        except Exception as e:
            logger.error(f"Failed to create client: {str(e)}")
            raise
    
    def get_client_by_email(self, email: str) -> Optional[SQLClient]:
        """Get client by email."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id, company_name, contact_name, contact_email,
                           subscription_tier, api_quota, api_usage, created_at, is_active
                    FROM clients WHERE contact_email = ? AND is_active = 1
                """, (email,))
                
                row = cursor.fetchone()
                if row:
                    return SQLClient(
                        client_id=row[0],
                        company_name=row[1],
                        contact_name=row[2],
                        contact_email=row[3],
                        subscription_tier=row[4],
                        api_quota=row[5],
                        api_usage=row[6],
                        created_at=row[7],
                        is_active=row[8]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get client by email: {str(e)}")
            return None
    
    def get_client_by_id(self, client_id: str) -> Optional[SQLClient]:
        """Get client by ID."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id, company_name, contact_name, contact_email,
                           subscription_tier, api_quota, api_usage, created_at, is_active
                    FROM clients WHERE client_id = ? AND is_active = 1
                """, (client_id,))
                
                row = cursor.fetchone()
                if row:
                    return SQLClient(
                        client_id=row[0],
                        company_name=row[1],
                        contact_name=row[2],
                        contact_email=row[3],
                        subscription_tier=row[4],
                        api_quota=row[5],
                        api_usage=row[6],
                        created_at=row[7],
                        is_active=row[8]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get client by ID: {str(e)}")
            return None
    
    def authenticate_client(self, email: str, password: str) -> Optional[SQLClient]:
        """Authenticate client with email and password."""
        try:
            client = self.get_client_by_email(email)
            if not client:
                return None
            
            # Get stored password hash
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT password_hash FROM clients WHERE client_id = ?
                """, (client.client_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                stored_hash = row[0]
                input_hash = self._hash_password(password)
                
                if stored_hash == input_hash:
                    return client
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return None
    
    def create_session(self, client_id: str) -> str:
        """Create a new session for client."""
        try:
            session_id = str(uuid.uuid4())
            
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (session_id, client_id, created_at, last_activity)
                    VALUES (?, ?, GETDATE(), GETDATE())
                """, (session_id, client_id))
                conn.commit()
            
            # Store in memory for quick access
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow()
            }
            
            logger.info(f"Session created: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
    
    def validate_session(self, session_id: str) -> Optional[SQLClient]:
        """Validate session and return client."""
        try:
            # Check memory first
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                client_id = session_data['client_id']
                
                # Update last activity
                session_data['last_activity'] = datetime.utcnow()
                
                # Get client from database
                return self.get_client_by_id(client_id)
            
            # Check database
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.client_id, s.created_at, s.last_activity
                    FROM sessions s
                    JOIN clients c ON s.client_id = c.client_id
                    WHERE s.session_id = ? AND s.is_active = 1 AND c.is_active = 1
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    client_id, created_at, last_activity = row
                    
                    # Check if session is not expired (24 hours)
                    if datetime.utcnow() - last_activity > timedelta(hours=24):
                        self.revoke_session(session_id)
                        return None
                    
                    # Update last activity
                    cursor.execute("""
                        UPDATE sessions SET last_activity = GETDATE() WHERE session_id = ?
                    """, (session_id,))
                    conn.commit()
                    
                    return self.get_client_by_id(client_id)
                
                return None
                
        except Exception as e:
            logger.error(f"Session validation failed: {str(e)}")
            return None
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        try:
            # Remove from memory
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Mark as inactive in database
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions SET is_active = 0 WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            
            logger.info(f"Session revoked: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke session: {str(e)}")
            return False
    
    def update_api_usage(self, client_id: str) -> bool:
        """Update API usage for client."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE clients SET api_usage = api_usage + 1, updated_at = GETDATE()
                    WHERE client_id = ?
                """, (client_id,))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update API usage: {str(e)}")
            return False
    
    def log_analytics(self, client_id: str, video_id: str, query: str, 
                     response_time: float, confidence_score: float) -> bool:
        """Log analytics data."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                # Log query
                cursor.execute("""
                    INSERT INTO queries (client_id, video_id, query_text, response_time, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (client_id, video_id, query, response_time, confidence_score))
                
                # Log analytics event
                cursor.execute("""
                    INSERT INTO analytics (client_id, event_type, event_data)
                    VALUES (?, ?, ?)
                """, (client_id, 'query_executed', f'{{"video_id": "{video_id}", "response_time": {response_time}}}'))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log analytics: {str(e)}")
            return False
    
    def get_client_analytics(self, client_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get client analytics data."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(response_time) as avg_response_time,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(DISTINCT video_id) as unique_videos
                    FROM queries 
                    WHERE client_id = ? AND created_at >= DATEADD(day, -?, GETDATE())
                """, (client_id, days))
                
                row = cursor.fetchone()
                if row:
                    return [{
                        'total_queries': row[0] or 0,
                        'avg_response_time': float(row[1]) if row[1] else 0,
                        'avg_confidence': float(row[2]) if row[2] else 0,
                        'unique_videos': row[3] or 0
                    }]
                
                return []
                
        except Exception as e:
            logger.error(f"Failed to get analytics: {str(e)}")
            return []
    
    def get_all_clients(self) -> List[SQLClient]:
        """Get all active clients."""
        try:
            with self.db.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT client_id, company_name, contact_name, contact_email,
                           subscription_tier, api_quota, api_usage, created_at, is_active
                    FROM clients WHERE is_active = 1
                    ORDER BY created_at DESC
                """)
                
                clients = []
                for row in cursor.fetchall():
                    clients.append(SQLClient(
                        client_id=row[0],
                        company_name=row[1],
                        contact_name=row[2],
                        contact_email=row[3],
                        subscription_tier=row[4],
                        api_quota=row[5],
                        api_usage=row[6],
                        created_at=row[7],
                        is_active=row[8]
                    ))
                
                return clients
                
        except Exception as e:
            logger.error(f"Failed to get all clients: {str(e)}")
            return []