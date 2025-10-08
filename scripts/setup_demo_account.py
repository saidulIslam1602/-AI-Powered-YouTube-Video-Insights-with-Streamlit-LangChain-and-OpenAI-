#!/usr/bin/env python3
"""Setup demo account for testing."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.client_manager import ClientManager

def setup_demo_account():
    """Create a demo account for testing."""
    print("Setting up demo account...")
    
    try:
        # Initialize client manager
        client_manager = ClientManager()
        
        # Create demo client
        client_id = client_manager.create_client(
            company_name="Demo Company",
            contact_email="demo@company.com",
            contact_name="Demo User",
            password="demo123",
            subscription_tier="professional"
        )
        
        print(f"Demo account created successfully!")
        print(f"Email: demo@company.com")
        print(f"Password: demo123")
        print(f"Company: Demo Company")
        print(f"User: Demo User")
        print(f"Tier: Professional")
        
    except ValueError as e:
        if "already exists" in str(e):
            print("Demo account already exists!")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    setup_demo_account()