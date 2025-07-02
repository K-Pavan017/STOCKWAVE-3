import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ABC'  # Replace with a strong, actual secret key
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'  # Use your preferred database URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False