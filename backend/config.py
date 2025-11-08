import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ABC'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgres:Pavan%40017@localhost:5432/db'

    SQLALCHEMY_TRACK_MODIFICATIONS = False
