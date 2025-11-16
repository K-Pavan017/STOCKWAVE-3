import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ABC'
    # Using your provided PostgreSQL URI
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgres:Pavan%%40017@localhost:5432/db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Configuration for Model Persistence (CRITICAL for prediction speed) ---
    # This directory is where the trained models and their scalers will be saved.
    MODEL_STORAGE_DIR = 'trained_models'
    
    # Check if the directory exists, and create it if it doesn't.
    # This prevents runtime errors when the prediction service tries to save models.
    if not os.path.exists(MODEL_STORAGE_DIR):
        os.makedirs(MODEL_STORAGE_DIR)
