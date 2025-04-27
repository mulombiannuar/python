from flask import request
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from application.services.user_service import UserService
from application.forms.user_form import RegistrationForm, LoginForm

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET'])
def login_page():
    form = RegistrationForm()
    return render_template('auth/login.html', form=form)


@auth.route('/login', methods=['POST'])
def login_handler():
    form = LoginForm()  
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        remember_me = form.remember.data  

        user = UserService.get_user_by_email(email)
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=remember_me)
                return redirect(url_for('home.dashboard_page'))
            else:
                flash('Incorrect password, try again.', category='danger') 
        else:
            flash('Email address does not exist.', category='danger') 
            
    return render_template('auth/login.html', form=form)


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
@login_required
def logout_handler():
    logout_user()
    flash('You are logged out successfully!', category='success')
    return redirect(url_for('auth.login_page'))