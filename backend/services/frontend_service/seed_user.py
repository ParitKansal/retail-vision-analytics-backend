import sys
import os

# Set up environment
sys.path.append('/workspace/ml-project')

from app import create_app
from app.models_mongo import User

app = create_app()

with app.app_context():
    email = os.getenv("SEED_USER_EMAIL", "demo@example.com")
    password = os.getenv("SEED_USER_PASSWORD")
    if not password:
        raise ValueError("SEED_USER_PASSWORD environment variable is required")
    
    existing_user = User.find_by_email(email)
    if existing_user:
        print(f"User {email} already exists.")
        
        # Optional: Update password just in case
        existing_user.set_password(password)
        existing_user.save()
        print("Password reset.")
    else:
        print(f"Creating user {email}...")
        user = User(
            email=email,
            password="temp", # set_password will hash it
            full_name="Demo User",
            role="admin"
        )
        user.set_password(password)
        user.save()
        print("User created successfully!")
