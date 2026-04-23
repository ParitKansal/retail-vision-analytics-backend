#!/usr/bin/env python3
"""
Backend startup script for Flask application
"""
import os
import sys

# Change to project directory and add to Python path
PROJECT_DIR = '/workspace/ml-project'
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Import Flask app
from app import create_app, db

# Create app instance
app = create_app()

# Initialize database
with app.app_context():
    print("Initializing database tables...")
    db.create_all()
    print("Database initialized successfully!")

# Run the application
if __name__ == "__main__":
    print("Starting Flask application on 0.0.0.0:5000")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
