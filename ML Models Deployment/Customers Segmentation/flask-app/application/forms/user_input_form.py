from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField
from wtforms.validators import DataRequired, NumberRange
from flask_wtf.file import FileAllowed

class UserInputForm(FlaskForm):
    annual_income_k = IntegerField('Annual Income (k$)', validators=[
        DataRequired(message="Annual Income is required."),
        NumberRange(min=0, max=137, message="Must be a positive number.")
    ])
    spending_score = IntegerField('Spending Score (0-100)', validators=[
        DataRequired(message="Spending Score is required."),
        NumberRange(min=0, max=100, message="Must be between 0 and 100.")
    ])
