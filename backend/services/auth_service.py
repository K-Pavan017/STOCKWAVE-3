from models.users import User
from database import db
from flask import jsonify

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

def login_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'email': user.email,
            'username': user.username
        }), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401