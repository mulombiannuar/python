from flask import Blueprint, render_template, request, flash, redirect, url_for
from application.models.user import User
from werkzeug.security import generate_password_hash, check_password_hash
from application import db   #means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET'])
def login_page():
    return render_template('auth/login.html')

@auth.route('/login', methods=['POST'])
def login_user():
    # Process login form submission
    pass

@auth.route('/register', methods=['GET'])
def register_page():
    return render_template('auth/register.html')

@auth.route('/register', methods=['POST'])
def register_user():
    # Process registration form submission
    pass

@auth.route('/forgot-password', methods=['GET'])
def forgot_password_page():
    return render_template('auth/forgot_password.html')

@auth.route('/forgot-password', methods=['POST'])
def send_password_reset_link():
    # Process sending password reset link
    pass

@auth.route('/reset-password/<token>', methods=['GET'])
def reset_password_page(token):
    return render_template('auth/reset_password.html', token=token)

@auth.route('/reset-password/<token>', methods=['POST'])
def reset_user_password(token):
    # Process resetting the password
    pass

@auth.route('/logout')
def logout_user():
    # Logout logic
    pass