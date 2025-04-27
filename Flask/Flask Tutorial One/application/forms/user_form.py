from flask_wtf import FlaskForm
from wtforms.validators import Email
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(message="Please enter your email address."),
        Email(message="Please enter a valid email address.")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message="Please enter your password."),
        Length(min=8, message="Password must be at least 8 characters long.")
    ])
    remember = BooleanField('Remember Me', default=False)



class RegistrationForm(FlaskForm):
    first_name = StringField('First Name', validators=[
        DataRequired(message="First name is required."),
        Length(min=2, max=150, message="First name must be between 2 and 150 characters.")
    ])
    last_name = StringField('Last Name', validators=[
        DataRequired(message="Last name is required."),
        Length(min=2, max=150, message="Last name must be between 2 and 150 characters.")
    ])
    email = StringField('Email', validators=[
        DataRequired(message="Email address is required."),
        Email(message="Please enter a valid email address.")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message="Password is required."),
        Length(min=8, message="Password must be at least 8 characters long.")
    ])
    address = StringField('Address', validators=[
        DataRequired(message="Address is required."),
        Length(min=10, max=255, message="Address must be between 10 and 255 characters.")
    ])
    zip_code = StringField('Zip Code', validators=[
        DataRequired(message="Zip code is required."),
        Length(min=5, max=20, message="Zip code must be between 5 and 20 characters.")
    ])
    gender = StringField('Gender', validators=[
        DataRequired(message="Please select your gender.")
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message="Please confirm your password."),
        EqualTo('password', message="Passwords must match.")
    ])
    agree = BooleanField('I agree to the terms', validators=[
        DataRequired(message="You must agree to the terms to register.")
    ])

