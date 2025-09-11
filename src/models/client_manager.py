"""Client management system for customer-facing features."""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import logger


@dataclass
class Client:
    """Client data structure."""
    client_id: str
    company_name: str
    contact_email: str
    contact_name: str
    subscription_tier: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    custom_branding: Optional[Dict[str, Any]] = None
    api_quota: int = 1000
    api_usage: int = 0


@dataclass
class ClientSession:
    """Client session data."""
    session_id: str
    client_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True


class ClientManager:
    """Manages client authentication, sessions, and data."""
    
    def __init__(self, db_path: str = "data/clients.db"):
        """Initialize client manager with database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the client database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    contact_email TEXT UNIQUE NOT NULL,
                    contact_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    subscription_tier TEXT DEFAULT 'basic',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    custom_branding TEXT,
                    api_quota INTEGER DEFAULT 1000,
                    api_usage INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_sessions (
                    session_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (client_id) REFERENCES clients (client_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    video_id TEXT,
                    query TEXT,
                    response_time REAL,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (client_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    session_id TEXT,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (client_id)
                )
            """)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.sha256((password + salt).encode()).hexdigest()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_value = password_hash.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == hash_value
        except ValueError:
            return False
    
    def create_client(self, company_name: str, contact_email: str, 
                     contact_name: str, password: str, 
                     subscription_tier: str = "basic") -> str:
        """Create a new client account."""
        client_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO clients (client_id, company_name, contact_email, 
                                       contact_name, password_hash, subscription_tier)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (client_id, company_name, contact_email, contact_name, 
                      password_hash, subscription_tier))
                conn.commit()
                logger.info(f"Created new client: {company_name} ({client_id})")
                return client_id
            except sqlite3.IntegrityError:
                raise ValueError("Email already exists")
    
    def authenticate_client(self, email: str, password: str) -> Optional[Client]:
        """Authenticate client and return client data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT client_id, company_name, contact_email, contact_name,
                       password_hash, subscription_tier, created_at, last_login,
                       is_active, custom_branding, api_quota, api_usage
                FROM clients WHERE contact_email = ? AND is_active = 1
            """, (email,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            if not self._verify_password(password, row[4]):
                return None
            
            # Update last login
            conn.execute("""
                UPDATE clients SET last_login = CURRENT_TIMESTAMP
                WHERE client_id = ?
            """, (row[0],))
            conn.commit()
            
            return Client(
                client_id=row[0],
                company_name=row[1],
                contact_email=row[2],
                contact_name=row[3],
                subscription_tier=row[5],
                created_at=datetime.fromisoformat(row[6]),
                last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                is_active=bool(row[8]),
                custom_branding=eval(row[9]) if row[9] else None,
                api_quota=row[10],
                api_usage=row[11]
            )
    
    def create_session(self, client_id: str, duration_hours: int = 24) -> str:
        """Create a new client session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO client_sessions (session_id, client_id, expires_at)
                VALUES (?, ?, ?)
            """, (session_id, client_id, expires_at))
            conn.commit()
        
        logger.info(f"Created session for client {client_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Client]:
        """Validate session and return client data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT c.client_id, c.company_name, c.contact_email, c.contact_name,
                       c.subscription_tier, c.created_at, c.last_login, c.is_active,
                       c.custom_branding, c.api_quota, c.api_usage
                FROM clients c
                JOIN client_sessions s ON c.client_id = s.client_id
                WHERE s.session_id = ? AND s.is_active = 1 
                AND s.expires_at > CURRENT_TIMESTAMP AND c.is_active = 1
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Client(
                client_id=row[0],
                company_name=row[1],
                contact_email=row[2],
                contact_name=row[3],
                subscription_tier=row[4],
                created_at=datetime.fromisoformat(row[5]),
                last_login=datetime.fromisoformat(row[6]) if row[6] else None,
                is_active=bool(row[7]),
                custom_branding=eval(row[8]) if row[8] else None,
                api_quota=row[9],
                api_usage=row[10]
            )
    
    def log_analytics(self, client_id: str, video_id: str, query: str, 
                     response_time: float, confidence_score: float):
        """Log client analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO client_analytics (client_id, video_id, query, 
                                            response_time, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            """, (client_id, video_id, query, response_time, confidence_score))
            conn.commit()
    
    def submit_feedback(self, client_id: str, session_id: str, rating: int, 
                       feedback_text: str = ""):
        """Submit client feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO client_feedback (client_id, session_id, rating, feedback_text)
                VALUES (?, ?, ?, ?)
            """, (client_id, session_id, rating, feedback_text))
            conn.commit()
    
    def get_client_analytics(self, client_id: str, days: int = 30) -> Dict[str, Any]:
        """Get client analytics for the specified period."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(response_time) as avg_response_time,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(DISTINCT video_id) as unique_videos
                FROM client_analytics 
                WHERE client_id = ? AND created_at >= datetime('now', '-{} days')
            """.format(days), (client_id,))
            
            row = cursor.fetchone()
            
            # Get daily usage
            cursor = conn.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as queries
                FROM client_analytics 
                WHERE client_id = ? AND created_at >= datetime('now', '-{} days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """.format(days), (client_id,))
            
            daily_usage = [{"date": row[0], "queries": row[1]} for row in cursor.fetchall()]
            
            return {
                "total_queries": row[0] or 0,
                "avg_response_time": row[1] or 0,
                "avg_confidence": row[2] or 0,
                "unique_videos": row[3] or 0,
                "daily_usage": daily_usage
            }
    
    def update_api_usage(self, client_id: str, usage: int = 1):
        """Update client API usage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE clients SET api_usage = api_usage + ? WHERE client_id = ?
            """, (usage, client_id))
            conn.commit()
    
    def is_quota_exceeded(self, client_id: str) -> bool:
        """Check if client has exceeded their API quota."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT api_usage, api_quota FROM clients WHERE client_id = ?
            """, (client_id,))
            row = cursor.fetchone()
            return row and row[0] >= row[1]