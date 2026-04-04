import os
from dotenv import load_dotenv

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ABC'
    # Using your provided PostgreSQL URI
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://stockdata_77iq_user:ovyMBXeabZI5nzm3vvDB1wNTmvA3uu6z@dpg-d6rq2q7afjfc73ejgi3g-a/stockdata_77iq'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,       # Test connections before use
        'pool_recycle': 300,         # Recycle connections every 5 min
        'pool_timeout': 20,
        'pool_size': 5,
        'max_overflow': 10,
    }


    # yfinance is used for data fetching and does not require an API key for basic usage.

    # --- Configuration for Model Persistence (CRITICAL for prediction speed) ---
    # This directory is where the trained models and their scalers will be saved.
    MODEL_STORAGE_DIR = 'trained_models'
    
    # Check if the directory exists, and create it if it doesn't.
    # This prevents runtime errors when the prediction service tries to save models.
    if not os.path.exists(MODEL_STORAGE_DIR):
        os.makedirs(MODEL_STORAGE_DIR)
