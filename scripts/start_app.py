#!/usr/bin/env python3
"""Start the YouTube Video Insights application."""

import subprocess
import sys
import os
import time

def start_streamlit():
    """Start the Streamlit application."""
    print("🚀 Starting YouTube Video Insights Application...")
    
    # Set environment variables
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Start Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--browser.gatherUsageStats', 'false',
        '--server.headless', 'true'
    ]
    
    try:
        print("📡 Starting server on http://localhost:8501")
        print("🌐 Also available on http://0.0.0.0:8501")
        print("⏳ Please wait for the application to start...")
        
        # Start the process
        process = subprocess.Popen(cmd, env=env)
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ Application started successfully!")
            print("🔗 Open your browser and go to: http://localhost:8501")
            print("🛑 Press Ctrl+C to stop the application")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping application...")
                process.terminate()
                process.wait()
                print("✅ Application stopped.")
        else:
            print("❌ Failed to start application. Check the logs.")
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    start_streamlit()