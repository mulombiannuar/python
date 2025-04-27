from flask import request
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user
from application.services.user_service import UserService
from application.forms.user_form import RegistrationForm, LoginForm

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
    form = RegistrationForm()
    return render_template('auth/register.html', form=form)


@auth.route('/register', methods=['GET', 'POST'])
def register_user():
    form = RegistrationForm()

    if form.validate_on_submit():  
        data = {
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'email': form.email.data,
            'password': form.password.data,
            'address': form.address.data,
            'zip_code': form.zip_code.data,
            'gender': form.gender.data
        }
        user = UserService.create_user(data)
        if user:
            flash(message='Your registration was successful. Please proceed to login!', category='success')
            return redirect(url_for('auth.login_page'))
        else:
            flash(message='Registration failed. Please try again.', category='danger')
            return redirect(url_for('auth.register_page'))
    
    return render_template('auth/register.html', form=form)
    

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