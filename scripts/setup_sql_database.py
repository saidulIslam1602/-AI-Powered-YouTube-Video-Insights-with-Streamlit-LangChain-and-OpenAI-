#!/usr/bin/env python3
"""Setup SQL Server database for YouTube Video Insights."""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.database.sql_server import db_manager
from src.models.sql_client_manager import SQLClientManager
from src.utils.logger import logger

def setup_database():
    """Setup the SQL Server database and create demo data."""
    try:
        print("🚀 Setting up Microsoft SQL Server database...")
        
        # Test connection
        if not db_manager.connection.test_connection():
            print("❌ Failed to connect to SQL Server")
            print("Make sure SQL Server is running on localhost:1435")
            return False
        
        print("✅ Connected to SQL Server successfully")
        
        # Create database and tables
        print("📊 Creating database and tables...")
        db_manager._ensure_database_exists()
        db_manager._create_tables()
        
        print("✅ Database and tables created successfully")
        
        # Create demo client
        print("👤 Creating demo client...")
        client_manager = SQLClientManager()
        
        # Check if demo client already exists
        demo_client = client_manager.get_client_by_email("demo@company.com")
        if not demo_client:
            client_id = client_manager.create_client(
                company_name="Demo Company Inc.",
                contact_email="demo@company.com",
                contact_name="Demo User",
                password="demo123",
                subscription_tier="professional"
            )
            print(f"✅ Demo client created with ID: {client_id}")
        else:
            print("✅ Demo client already exists")
        
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 Connection Details:")
        print(f"   Server: localhost:1435")
        print(f"   Database: YouTubeInsights")
        print(f"   Username: sa")
        print(f"   Password: YourStrong@Pass123")
        print("\n👤 Demo Account:")
        print(f"   Email: demo@company.com")
        print(f"   Password: demo123")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {str(e)}")
        logger.error(f"Database setup error: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)