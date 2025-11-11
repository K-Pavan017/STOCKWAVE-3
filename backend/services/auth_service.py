from models.users import User
from database import db
from datetime import datetime, timedelta
from flask import jsonify, make_response # 👈 ADDED make_response
import jwt # 👈 Assuming you use JWTs for stateless auth (Highly recommended)
import os # For getting your JWT secret key

# --- Configuration (Add this near the top) ---
# NOTE: Replace 'your_super_secret_key' with a strong key from your Config/Env Vars
SECRET_KEY = os.environ.get('SECRET_KEY', 'your_super_secret_key') 

# Assuming 7 days of login time
COOKIE_MAX_AGE = timedelta(days=7) 

def register_user(username, email, password):
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 409

    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True, 'message': 'User registered successfully'}), 201

# --- EDITED FUNCTION FOR CROSS-DOMAIN LOGIN ---
def login_user(email, password):
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        
        # 1. Create a token (JWT recommended for stateless API)
        # This payload identifies the user on subsequent requests
        payload = {
            'user_id': user.id,
            'username': user.username,
            # Add an expiration time for the token
            'exp': datetime.utcnow() + COOKIE_MAX_AGE 
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

        # 2. Create the success response
        response_data = {
            'success': True,
            'message': 'Login successful',
            'email': user.email,
            'username': user.username
        }
        # Use make_response to allow setting a cookie on the HTTP response
        response = make_response(jsonify(response_data), 200)

        # 3. 🍪 Set the cookie with CRITICAL cross-domain flags
        response.set_cookie(
            'auth_token', 
            token, 
            httponly=True,       # Prevents client-side JavaScript access (security)
            secure=True,         # ⭐ CRITICAL: Must be True for HTTPS (Vercel/Render)
            samesite='None',     # ⭐ CRITICAL: Allows cookie to be sent cross-domain
            max_age=COOKIE_MAX_AGE # Sets the cookie expiration time
        )

        return response
    
    else:
        # Invalid credentials response
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401