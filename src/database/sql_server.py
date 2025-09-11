"""Microsoft SQL Server database integration."""

import pyodbc
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime
import logging

try:
    from src.config.settings import settings
    from src.utils.logger import logger
except ImportError:
    # Fallback for direct execution
    import logging
    logger = logging.getLogger(__name__)
    settings = None

class SQLServerConnection:
    """Microsoft SQL Server database connection manager."""
    
    def __init__(self):
        self.server = os.getenv('SQL_SERVER_HOST', 'localhost')
        self.port = os.getenv('SQL_SERVER_PORT', '1435')
        self.database = os.getenv('SQL_SERVER_DATABASE', 'YouTubeInsights')
        self.username = os.getenv('SQL_SERVER_USERNAME', 'sa')
        self.password = os.getenv('SQL_SERVER_PASSWORD', 'YourStrong@Pass123')
        self.driver = '{ODBC Driver 18 for SQL Server}'
        
    def get_connection_string(self) -> str:
        """Get SQL Server connection string."""
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
            f"Encrypt=yes;"
        )
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = None
        try:
            conn = pyodbc.connect(self.get_connection_string())
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def create_database(self) -> bool:
        """Create the YouTubeInsights database if it doesn't exist."""
        try:
            # Connect to master database first
            master_conn_str = (
                f"DRIVER={self.driver};"
                f"SERVER={self.server},{self.port};"
                f"DATABASE=master;"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"TrustServerCertificate=yes;"
                f"Encrypt=yes;"
            )
            
            with pyodbc.connect(master_conn_str, autocommit=True) as conn:
                cursor = conn.cursor()
                
                # Check if database exists
                cursor.execute(f"SELECT name FROM sys.databases WHERE name = '{self.database}'")
                if not cursor.fetchone():
                    # Create database
                    cursor.execute(f"CREATE DATABASE [{self.database}]")
                    logger.info(f"Database '{self.database}' created successfully")
                else:
                    logger.info(f"Database '{self.database}' already exists")
                
                return True
        except Exception as e:
            logger.error(f"Failed to create database: {str(e)}")
            return False

class DatabaseManager:
    """Database operations manager for YouTube Insights."""
    
    def __init__(self):
        self.connection = SQLServerConnection()
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure the database exists."""
        if not self.connection.create_database():
            raise Exception("Failed to create or access database")
    
    def _create_tables(self):
        """Create all required tables."""
        tables = [
            self._create_clients_table(),
            self._create_sessions_table(),
            self._create_videos_table(),
            self._create_queries_table(),
            self._create_analytics_table(),
            self._create_feedback_table()
        ]
        
        for table_name in tables:
            if table_name:
                logger.info(f"Table '{table_name}' created or verified")
    
    def _create_clients_table(self) -> str:
        """Create clients table."""
        table_name = "clients"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    client_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
                    company_name NVARCHAR(255) NOT NULL,
                    contact_name NVARCHAR(255) NOT NULL,
                    contact_email NVARCHAR(255) UNIQUE NOT NULL,
                    password_hash NVARCHAR(255) NOT NULL,
                    subscription_tier NVARCHAR(50) NOT NULL DEFAULT 'basic',
                    api_quota INT NOT NULL DEFAULT 1000,
                    api_usage INT NOT NULL DEFAULT 0,
                    created_at DATETIME2 DEFAULT GETDATE(),
                    updated_at DATETIME2 DEFAULT GETDATE(),
                    is_active BIT DEFAULT 1
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None
    
    def _create_sessions_table(self) -> str:
        """Create sessions table."""
        table_name = "sessions"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    session_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
                    client_id UNIQUEIDENTIFIER NOT NULL,
                    created_at DATETIME2 DEFAULT GETDATE(),
                    last_activity DATETIME2 DEFAULT GETDATE(),
                    is_active BIT DEFAULT 1,
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None
    
    def _create_videos_table(self) -> str:
        """Create videos table."""
        table_name = "videos"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    video_id NVARCHAR(255) PRIMARY KEY,
                    title NVARCHAR(500) NOT NULL,
                    description NVARCHAR(MAX),
                    duration INT,
                    language NVARCHAR(10),
                    transcript_text NVARCHAR(MAX),
                    processed_at DATETIME2 DEFAULT GETDATE(),
                    client_id UNIQUEIDENTIFIER,
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE SET NULL
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None
    
    def _create_queries_table(self) -> str:
        """Create queries table."""
        table_name = "queries"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    query_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
                    client_id UNIQUEIDENTIFIER NOT NULL,
                    video_id NVARCHAR(255),
                    query_text NVARCHAR(MAX) NOT NULL,
                    response_text NVARCHAR(MAX),
                    confidence_score FLOAT,
                    response_time FLOAT,
                    created_at DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE SET NULL
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None
    
    def _create_analytics_table(self) -> str:
        """Create analytics table."""
        table_name = "analytics"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    analytics_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
                    client_id UNIQUEIDENTIFIER NOT NULL,
                    event_type NVARCHAR(100) NOT NULL,
                    event_data NVARCHAR(MAX),
                    created_at DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None
    
    def _create_feedback_table(self) -> str:
        """Create feedback table."""
        table_name = "feedback"
        try:
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                CREATE TABLE {table_name} (
                    feedback_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
                    client_id UNIQUEIDENTIFIER NOT NULL,
                    query_id UNIQUEIDENTIFIER,
                    rating INT CHECK (rating >= 1 AND rating <= 5),
                    feedback_text NVARCHAR(MAX),
                    created_at DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE,
                    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE SET NULL
                )
                """)
                conn.commit()
                return table_name
        except Exception as e:
            logger.error(f"Failed to create {table_name} table: {str(e)}")
            return None

# Global database manager instance
db_manager = DatabaseManager()